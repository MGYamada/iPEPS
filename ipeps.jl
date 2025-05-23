using LinearAlgebra
using OMEinsum
using Zygote
using Optim
using KrylovKit

const dim = 2

Zygote.@adjoint LinearAlgebra.svd(A) = svd_back(A)

function svd_back(A; η = 1e-40)
    U, S, V = svd(A)
    (U, S, V), function (Δ)
        ΔA = Δ[2] === nothing ? zeros(eltype(A), size(A)...) : U * Diagonal(Δ[2]) * V'
        if Δ[1] !== nothing || Δ[3] !== nothing
            S² = S .^ 2
            invS = @. S / (S² + η)
            F = S²' .- S²
            @. F /= (F ^ 2 + η)
            if Δ[1] !== nothing
                J = F .* (U' * Δ[1])
                ΔA .+= U * (J .+ J') * Diagonal(S) * V'
                ΔA .+= (I - U * U') * Δ[1] * Diagonal(invS) * V'
            end
            if Δ[3] !== nothing
                K = F .* (V' * Δ[3])
                ΔA .+= U * Diagonal(S) * (K .+ K') * V'
                L = Diagonal(diag(V' * Δ[3]))
                ΔA .+= 0.5 .* U * Diagonal(invS) * (L' .- L) * V'
                ΔA .+= U * Diagonal(invS) * Δ[3]' * (I - V * V')
            end
        end
        (ΔA,)
    end
end

function symmetrize(x)
    x += permutedims(x, (1, 4, 3, 2, 5)) # left-right
    x += permutedims(x, (3, 2, 1, 4, 5)) # up-down
    x += permutedims(x, (2, 1, 4, 3, 5)) # diagonal
    x += permutedims(x, (4, 3, 2, 1, 5)) # rotation
    x / norm(x)
end

function initialize(a, χ)
    corner = randn(χ, χ)
    edge = randn(χ, size(a, 1), χ)
    corner += transpose(corner)
    edge += ein"ijk -> kji"(edge)
    corner, edge
end

function step(rt, a, χ, D)
    corner, edge = rt
    # growx
    cp = ein"((ad, iba), dcl), jkcb -> ijlk"(corner, edge, edge, a) # fix later
    tp = ein"iam, jkla -> ijklm"(edge, a)

    # renormalize
    cpmat = reshape(cp, χ * D, χ * D)
    cpmat += cpmat'
    u, s, v = svd(cpmat)
    z = reshape(u[:, 1 : χ], χ, D, χ)

    corner = ein"(abcd, abi), cdj -> ij"(cp, conj.(z), z)
    edge = ein"(abjcd, abi), dck -> ijk"(tp, conj.(z), z)

    # indexperm_symmetrize
    corner += transpose(corner)
    edge += ein"ijk -> kji"(edge)

    # normalize
    corner /= norm(corner)
    edge /= norm(edge)

    corner, edge
end

function fixedpointbackward(next, g, args...)
    _, back = pullback(next, g, args...)
    back1 = x -> back(x)[1]
    back2 = x -> back(x)[2 : end]
    function (Δ)
        grad, = linsolve(Δ; ishermitian = false) do x
            x .- back1(x)
        end
        back2(grad)
    end
end

function ctmrg(a, rt, χ, D; tol = 1e-12, maxit = 100)
    for i in 1 : maxit
        rtold = rt
        rt = step(rt, a, χ, D) # gauge fix
        println(norm.(rt .- rtold))
    end
    rt
end

@Zygote.adjoint function ctmrg(a, rt, χ, D; args...)
    rt = ctmrg(a, rt, χ, D; args...)
    rt, Δ -> (nothing, nothing, fixedpointbackward(step, rt, a, χ, D)(Δ)...)
end

function energy(h, ipeps, χ; tol = 1e-12, maxit = 100)
    D = size(ipeps, 1) ^ 2
    s = size(ipeps, 5)
    ipeps = symmetrize(ipeps)
    ap = ein"abcdx, ijkly -> aibjckdlxy"(ipeps, conj.(ipeps))
    ap = reshape(ap, D, D, D, D, s, s)
    a = ein"ijklaa -> ijkl"(ap)

    rt = initialize(a, χ)
    rt = ctmrg(a, rt, χ, D; tol = tol, maxit = maxit)
    expectationvalue(h, ap, rt)
end

function expectationvalue(h, ap, rt)
    corner, edge = rt
    ap /= norm(ap)
    l = ein"ab, (ica, (bde, (cjfdlm, (eg, gfk)))) -> ijklm"(corner, edge, edge, ap, corner, edge) # fix later
    e = ein"(abcij, abckl), ijkl -> "(l, l, h)[]
    n = ein"ijkaa, ijkbb -> "(l, l)[]
    e / n
end

function vipeps(ipeps, h; χ = 30, tol = 1e-12, f_tol = 1e-8, maxit = 100)
    f(x) = real(energy(h, x, χ; tol = tol, maxit = maxit))
    res = optimize(f, f', ipeps, LBFGS(), Optim.Options(show_trace = true, f_tol = f_tol); inplace = false)
    println(res)
    println("energy: ", minimum(res))
end

function main()
    ipeps = symmetrize(randn(dim, dim, dim, dim, 2))
    σx = [0.0 1.0; 1.0 0.0]
    σy = [0.0 -1.0im; 1.0im 0.0]
    σz = [1.0 0.0; 0.0 -1.0]
    h = ein"ij, kl -> ijkl"(σz, σz) .- ein"ij, kl -> ijkl"(σx, σx) .- ein"ij, kl -> ijkl"(σy, σy)
    h = ein"ijcd, kc, ld -> ijkl"(h, σx, σx')
    vipeps(ipeps, h ./ 2)
end

main()
