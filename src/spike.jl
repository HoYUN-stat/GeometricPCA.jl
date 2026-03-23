"""
    generate_spiked_data(n::Int, p::Int, signal::Float64; noise_var::Float64)

CDF of the standard normal distribution, i.e., 
```math
\\Phi(x) = \\mathbb{P}(Z \\le x) = \\frac{1}{2} \\left( 1 + \\operatorname{erf}\\left( \\frac{x}{\\sqrt{2}} \\right) \\right).
```

```jldoctest
julia> abc
```
"""
function generate_spiked_data(n::Int, p::Int, signal::Vector{Float64}; noise_var::Float64)
    Z = randn(p, true_d)
    V_true = Matrix(qr(Z).Q)
    Signals = randn(n, true_d) .* sqrt(signal_strength)
    X = Signals * V_true'
    X .+= randn(n, p) .* sqrt(noise_var)
    return X
end