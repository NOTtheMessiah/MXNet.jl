using ArgParse
using Images
using MXNet

parse_command_line() = begin
    s = ArgParseSettings(description = "Run Neural-Style on an immage")
    @add_arg_table s begin
        "--model", "-m"
            help = "which neural network to use"
            default = "vgg19"
        "--content-image"
            arg_type = AbstractString 
            help = "the content image to reproduce"
            default = "input/IMG_4343.jpg"
        "--style-image"
            arg_type = AbstractString
            help = "image to determine the visual style"
            default="input/starry_night.jpg"
            help="the style image"
        "--stop-eps"
            arg_type = Float64
            help="stop when the relative change is less than this"
            default=.005
        "--content-weight"
            arg_type = Float64
            default=10
            help="the weight for the content image"
        "--style-weight"
            arg_type = Float64
            default=1
            help="the weight for the style image"
        "--max-num-epochs"
            arg_type = Int
            default=1000
            help="the maximal number of training epochs"
        "--max-long-edge"
            arg_type = Int
            default=600
            help="resize the content image"
        "--lr"
            arg_type = Float64
            default=.1
            help="the initial learning rate"
        "--gpu"
            arg_type = Int
            default=0
            help="which gpu card to use, -1 means using cpu"
        "--output"
            arg_type = AbstractString
            default="output/out.jpg"
            help="the output image"
        "--save-epochs"
            arg_type = Int
            default=50
            help="save the output every n epochs"
        "--remove-noise"
            arg_type = Float64
            default=0.2
            help="the magnitude to remove noise (unimplemented)"
    end
    parse_args(s)
end

args = parse_command_line()

function preprocess_content_image(path::AbstractString, longEdge::Int)
    img = load(path)
    println("load the content image, size = $(img.data |> size)")
    factor = (longEdge) / (img.data |> size |> x -> max(x...))
    new_size = map( x -> Int(floor(factor*x)) , img.data |> size)
    resized_img = Images.imresize(img, new_size)
    sample = separate(resized_img).data * 256
    # sub mean
    sample[:,:,1] -= 123.68
    sample[:,:,2] -= 116.779
    sample[:,:,3] -= 103.939
    println("resize the content image to $(new_size)")
    return reshape(sample, (size(sample)[1], size(sample)[2], 3, 1))
end


function preprocess_style_image(path::AbstractString, shape)
    img = load(path)
    resized_img = Images.imresize(img, (shape[2], shape[1]))
    sample = separate(resized_img).data * 256
    sample[:,:,1] -= 123.68
    sample[:,:,2] -= 116.779
    sample[:,:,3] -= 103.939
    return reshape(sample, (size(sample)[1], size(sample)[2], 3, 1))
end

function postprocess_image(img)
    img = reshape(img, (size(img)[1], size(img)[2], 3))
    img[:,:,1] += 123.68
    img[:,:,2] += 116.779
    img[:,:,3] += 103.939
    img = clamp(img, 0, 255)
    return map(UInt8,(img |> floor))
end

function save_image(img::Array{Float32,4}, filename::AbstractString)
    println("save output to $filename")
    println("dimensions are $(img|>size)")
    out = postprocess_image(img)
    #out = denoise_tv_chambolle(out, weight=args.remove_noise, multichannel=True)
    save(filename, colorim(out))
end

# assumes there's only one julia process using the GPU
getmem() = pipeline(`nvidia-smi`,`grep julia`) |> readall |> split |> x->x[end-1]

#input
args["gpu"] |> println

dev = args["gpu"] >= 0 ? mx.gpu(args["gpu"]) : mx.cpu()
content_np = preprocess_content_image(args["content-image"], args["max-long-edge"])
style_np = preprocess_style_image(args["style-image"], content_np|> size)
shape = size(content_np)[1:3]

#model
type SGExecutor
    executor
    data
    data_grad
end

function style_gram_executor(input_shape, ctx)
    # symbol
    data = mx.Variable("conv")
    rs_data = mx.Reshape(data=data, target_shape=(Int(prod(input_shape[1:2])),Int(input_shape[3]) ))
    weight = mx.Variable("weight")
    rs_weight = mx.Reshape(data=weight, target_shape=(Int(prod(input_shape[1:2])),Int(input_shape[3]) ))
    fc = mx.FullyConnected(data=rs_data, weight=rs_weight, no_bias=true, num_hidden=input_shape[3])
    # executor
    conv = mx.zeros(input_shape, ctx)
    grada = mx.zeros(input_shape, ctx)
    args = Dict(:conv => conv, :weight => conv)
    grad = Dict(:conv => grada)
    reqs = Dict(:conv => mx.GRAD_WRITE, :weight => mx.GRAD_NOP )
    executor = mx.bind(fc, ctx, args, args_grad=grad, grad_req=reqs)
    return SGExecutor(executor, conv, grad[:conv])
end

include("model_$(args["model"]).jl")

model_executor = get_model(shape, dev)
gram_executor = [style_gram_executor(arr |> size, dev) for arr in model_executor.style]

# get style representation
style_array = [mx.zeros(gram.executor.outputs[1] |> size, dev) for gram in gram_executor]
model_executor.data[:] = style_np
mx.forward(model_executor.executor)

for i in 1:length(model_executor.style)
    copy!(gram_executor[i].data,model_executor.style[i])
    mx.forward( gram_executor[i].executor )
    copy!(style_array[i],gram_executor[i].executor.outputs[1])
end

# get content representation
content_array = mx.zeros(model_executor.content |> size, dev)
content_grad  = mx.zeros(model_executor.content |> size, dev)
model_executor.data[:] = content_np
mx.forward(model_executor.executor)
copy!(content_array,model_executor.content)

 # train
img = mx.zeros(content_np |> size, dev)
img[:] = mx.rand(-0.1, 0.1, img |> size)

#= added to mx.LearningRate in optimizer.jl

type Factor <: AbstractLearningRateScheduler
    step :: Int
    factor :: Real
    learning_rate :: Real
end

get_learning_rate(self :: Factor, state :: OptimizationState ) =
    self.learning_rate * self.factor ^ ( state.curr_iter // self.step )
=#

lr = mx.LearningRate.Factor(10, .9, args["lr"])

optimizer = mx.SGD(
    lr = args["lr"],
    momentum = 0.9,
    weight_decay = 0.005,
    lr_scheduler = lr,
    grad_clip = 10)
optim_state = mx.create_state(optimizer,0, img)
optimizer.state = mx.OptimizationState(10)

println("start training arguments $args")
old_img = img |> copy
new_img = old_img

for epoch in 1:args["max-num-epochs"]
    copy!(model_executor.data,img  )
    mx.forward(model_executor.executor)

    # style gradient
    for i in 1:length(model_executor.style)
        copy!(gram_executor[i].data,model_executor.style[i])
        mx.forward(gram_executor[i].executor)
        mx.backward(gram_executor[i].executor,[gram_executor[i].executor.outputs[1] - style_array[i]])
        mx.div_from!(gram_executor[i].data_grad, (size(gram_executor[i].data)[3] ^2) * (prod(size(gram_executor[i].data)[1:2])))
        mx.mul_to!(gram_executor[i].data_grad, args["style-weight"])
    end

    # content gradient
    mec = model_executor.content |> copy
    @mx.nd_as_jl ro=content_array rw=content_grad begin content_grad[:] = (mec - content_array) * args["content-weight"] end

    # image gradient
    grad_array = append!([gram_executor[i].data_grad::mx.NDArray for i in 1:length(gram_executor)] , [content_grad::mx.NDArray])
    mx.backward(model_executor.executor,grad_array)

    mx.update(optimizer,0, img, model_executor.data_grad, optim_state)

    new_img = img |> copy
    eps = vecnorm(old_img - new_img) / vecnorm(new_img)
    old_img = new_img
    println("epoch $epoch, $(args["gpu"] >= 0 ? "GPU RAM $(getmem()), ": "")relative change $eps")

    if eps < args["stop-eps"]
        println("eps < $(args["stop-eps"]), training finished")
        break
    end

    if (epoch+1) % args["save-epochs"] == 0
        save_image(new_img, "output/tmp_$(string(epoch+1)).jpg")
    end
end

save_image(new_img, args["output"])
