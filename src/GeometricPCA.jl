module GeometricPCA

using LinearAlgebra
using Statistics
using Dates
using Random

include("spike.jl")
include("semiortho.jl")

export generate_spiked_data, generate_spiked_data!
export SemiOrthogonal, TallSkinnyIdentity, schatten_norm_pow, geometric_risk, geometric_risk_sequential



end
