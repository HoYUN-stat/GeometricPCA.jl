module GeometricPCA

using LinearAlgebra
using Statistics
using Dates
using Random

include("spike.jl")
include("semiortho.jl")
include("phasetrans.jl")
include("adaptiveqr.jl")

export generate_spiked_data, generate_spiked_data!
export SemiOrthogonal, TallSkinnyIdentity, schatten_norm_pow, geometric_risk, geometric_risk_sequential
export weight_bbp, upper_idx, signal_thres, limit_geometric_risk, lower_idx
export PCAWorkspace, update_subspace_optimized!, BlockPCAWorkspace, update_subspace_block!

end
