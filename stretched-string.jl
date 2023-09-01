using HDF5
using LinearAlgebra
using BandedMatrices
using IntelITT

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

function compute_determinants(σ_range, c, Δx, nsteps)
    A = BandedMatrix{Float64}(Zeros((nsteps + 1) * 2, (nsteps + 1) * 2), (2, 1 + 2))

    minval = 9000
    min_σ = 0

    for σ in σ_range
        fill_matrix_direct!(A, c, σ, Δx, nsteps)

        LU = BandedMatrices.lu!(A)

        det = 1

        for i in 1:size(A, 1)
            @inbounds det *= BandedMatrices.inbands_getindex(LU.factors, i, i)
        end
        
        for i in 1:size(A, 1)
            if i != @inbounds LU.ipiv[i]
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


compute_determinants(range(0.6,0.6,1), 0.2, 0.1, 10)

IntelITT.resume()

compute_determinants(range(0.6,0.7,10001), 0.2, 0.0001, 10000)

IntelITT.pause()
