using HDF5
using LinearAlgebra
using BandedMatrices

@inline function construct_jacobian(c, σ)
    A = Matrix{Float64}(Zeros(2, 2))

    A[2, 1] = -σ^2 / c^2
    A[1, 2] = 1

    return A
end


function construct_matrix(c, σ, Δx, nsteps)
    A = BandedMatrix{Float64}(Zeros((nsteps + 1) * 2, (nsteps + 1) * 2), (2, 1))

    A[1,1] = 1

    for i in 1:nsteps
        coord1 = 2 + (i - 1) * 2
        coord2 = coord1 - 1
   
        A[coord1:coord1 + 1, coord2:coord2 + 1] .= 1.0I + Δx * construct_jacobian(c, σ)
        A[coord1:coord1 + 1, coord2 + 2:coord2 + 3] .= -1.0 * Matrix(I, 2, 2)
    end

    A[nsteps * 2 + 2, nsteps * 2 + 1] = 1

    A
end

function fill_matrix_direct_dumb!(A, c, σ, Δx, nsteps)
    A[1,1] = 1

    for i in 1:nsteps
        coord1 = 2 + (i - 1) * 2
        coord2 = coord1 - 1
    
        A[coord1, coord2] = 1
        A[coord1, coord2 + 1] = Δx
        A[coord1 + 1, coord2] = - Δx * σ^2 / c^2
        A[coord1 + 1, coord2 + 1] = 1
        A[coord1, coord2 + 2] = -1
        A[coord1 + 1, coord2 + 3] = -1
    end

    A[nsteps * 2 + 2, nsteps * 2 + 1] = 1

    A
end
function fill_matrix_direct!(A, c, σ, Δx, nsteps)
    A[1,1] = 1
    A[1,2] = 0

    for i in 1:nsteps
        coord1 = 2 + (i - 1) * 2
        coord2 = coord1 - 1
    
        @inbounds BandedMatrices.inbands_setindex!(A, 1, coord1, coord2)
        @inbounds BandedMatrices.inbands_setindex!(A, Δx, coord1, coord2 + 1)
        @inbounds BandedMatrices.inbands_setindex!(A, - Δx * σ^2 / c^2, coord1 + 1, coord2)
        @inbounds BandedMatrices.inbands_setindex!(A, 1, coord1 + 1, coord2 + 1)
        @inbounds BandedMatrices.inbands_setindex!(A, 0, coord1 + 2, coord2 + 1)
        @inbounds BandedMatrices.inbands_setindex!(A, -1, coord1, coord2 + 2)
        @inbounds BandedMatrices.inbands_setindex!(A, 0, coord1 + 1, coord2 + 2)
        @inbounds BandedMatrices.inbands_setindex!(A, -1, coord1 + 1, coord2 + 3)
    end

    A[nsteps * 2 + 2, nsteps * 2 + 1] = 1
    A[nsteps * 2 + 2, nsteps * 2 + 2] = 0

    A
end


function fill_matrix_direct_banded!(A, c, σ, Δx, nsteps)
   
    band1 = BandedMatrices.dataview(view(A, band(1)))
    band0 = BandedMatrices.dataview(view(A, band(0)))
    bandn1 = BandedMatrices.dataview(view(A, band(-1)))
    bandn2 = BandedMatrices.dataview(view(A, band(-2)))

    band1[1] = 0
    band0[1] = 1.0
    bandn1[1] = 1.0

    for i in 1:nsteps
        offset = i * 2

        band1[offset] = -1.0
        band1[offset + 1] = -1.0

        bandn1[offset] = 1.0
        bandn1[offset + 1] = 1.0

        band0[offset] = Δx

        bandn2[offset - 1] = - Δx * σ^2 / c^2
    end

    band0[nsteps + 1] = 0.0

    A
end


function compute_determinants(σ_range, c, Δx, nsteps)
    A = BandedMatrix{Float64}(Zeros((nsteps + 1) * 2, (nsteps + 1) * 2), (2, 1 + 2))

    minval = 9000
    min_σ = 0

    for σ in σ_range
        fill_matrix_direct!(A, c, σ, Δx, nsteps)

        LU = BandedMatrices.lu!(A)

        det = 1

        for i in 1:size(A, 1)
            det *= (@inbounds BandedMatrices.inbands_getindex(LU.factors, i, i))
        end
        
        for (i, m) in enumerate(LU.ipiv)
            if i != m
                det *= -1
            end
        end

        if abs(det) < minval
            minval = det
            min_σ = σ
        end
    end

    println("Minimum solution: $(min_σ) (det = $(minval))")
    println("Actual solution: $(π * c / Δx / nsteps )")
end
