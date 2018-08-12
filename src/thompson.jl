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
    ll
    lσ
    logNoise
    gp_iter
    x0
end

ThompsonSampler(f,X,Y,M,
                lower=fill(-Inf,size(X,1)),
                upper=fill(Inf,size(Y,1)),
                ll=zeros(size(X,1)),
                lσ=0.0,
                logNoise=-1.0,
                gp_iter=1,
                x0=X[:,findmax(Y)[2]]) = ThompsonSampler(f,
                                                         size(X,1),
                                                         X,
                                                         Y,                        
                                                         M,
                                                         fill(-Inf,size(X,1)),
                                                         fill(Inf,size(Y,1))
                                                         ll,
                                                         lσ,
                                                         logNoise,
                                                         gp_iter,
                                                         x0)

function step(t::ThompsonSampler)
    
    X = t.X
    Y = t.Y

    gp = GP(X,Y,MeanZero(),SE(t.ll,t.lσ),t.logNoise)
    optimize!(gp,iterations=t.gp_iter)

    f,g = spectral_sample(gp,t.M)

    β = optimize(f,g,t.lower,t.upper,t.x0,Fminbox())
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
