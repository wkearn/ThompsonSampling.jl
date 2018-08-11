include("thompson.jl")

x = linspace(0,2pi,100)
y = linspace(0,2pi,100)

g(x,y) = sin(x)*sin(y)

gη(x) = g(x[1],x[2]) + 0.05*randn()

x0 = [π 3π/2;π π+1.0]
y0 = [gη(x0[:,i]) for i in 1:size(x0,2)]

t = ThompsonSampler(gη,x0,y0,100,[0.0,0.0],[2pi,2pi])

collect(Iterators.take(t,2))
@time collect(Iterators.take(t,50));
