# Thompson sampling for Bayesian optimization of continuous functions

using GaussianProcesses, Optim, Distributions

mutable struct ThompsonSampler
    f
    dim
    X
    Y
    M
    lower
    upper
end

ThompsonSampler(f,X,Y,M) = ThompsonSampler(f,
                                           size(X,1),
                                           X,
                                           Y,                                          
                                           M,
                                           fill(-Inf,size(X,1)),
                                           fill(Inf,size(Y,1)))

ThompsonSampler(f,X,Y,M,lower,upper) = ThompsonSampler(f,
                                                       size(X,1),
                                                       X,
                                                       Y,                                                      
                                                       M,
                                                       lower,
                                                       upper)

function step(t::ThompsonSampler)
    X = t.X
    Y = t.Y

    gp = GP(X,Y,MeanZero(),SE(zeros(size(X,1)),0.0),-1.0)
    optimize!(gp,iterations=1)

    f,g = spectral_sample(gp,t.M)

    # Start from our best maximum point
    x0 = X[:,findmax(Y)[2]]

    β = optimize(f,g,t.lower,t.upper,x0,Fminbox())
    Optim.minimizer(β)
end

Base.start(t::ThompsonSampler) = t
function Base.next(t::ThompsonSampler,state)
    xn = step(t)
    append!(t,xn)
    
    xn,t
end
Base.done(t::ThompsonSampler,state) = false

function Base.append!(t::ThompsonSampler,xn)
    yn = t.f(xn) # This is the one call to the expensive function f

    t.X = hcat(t.X,xn)
    t.Y = vcat(t.Y,yn)
    
    t
end

Base.iteratorsize(::ThompsonSampler) = Base.IsInfinite()

function spectral_sample(gp,M)
    l2 = gp.k.iℓ2
    σ2 = gp.k.σ2
    σξ = exp(gp.logNoise)

    # This is the right probability density for the SE kernel
    dW=MvNormal(zeros(length(l2)),sqrt.(l2))
    db=Uniform(0,2π)

    W = rand(dW,M)
    b = rand(db,M)

    Φ = hcat([ϕ(gp.X[:,i],σ2,W,b) for i in 1:size(gp.X,2)]...)

    cK = GaussianProcesses.tolerant_PDMat(Φ*Φ' + σξ^2*I)

    μ = cK\(Φ*gp.y)
    
    θ = μ + σξ*inv(cK.chol[:U])*randn(length(μ))
    f = x->-ϕ(x,σ2,W,b)'θ # Minus sign because Optim will minimize this for us
    g = (G,x)->G[:] = -ϕ′(x,σ2,W,b)'θ
    f,g
end

ϕ(x,σ2,W,b) = sqrt(2*σ2/length(b)) * cos.(W'x + b)
ϕ′(x,σ2,W,b) = -sqrt(2*σ2/length(b)) * sin.(W'x + b).*W'
k(x,y,σ2,W,b) = 2*σ2/length(b)*cos.(W*x+b)'cos.(W*y+b)
