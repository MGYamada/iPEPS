using LinearAlgebra
using OMEinsum
using Zygote
using Optim

const dim = 2

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
    corner += corner'
    edge += permutedims(conj.(edge), (3, 2, 1))
    corner, edge
end

function ctmrg(a, rt, χ, D; tol = 1e-10, maxit = 100)
    corner, edge = rt
    valsold = zeros(χ * D)
    for i in 1 : maxit
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

        vals = s ./ s[1]

        # indexperm_symmetrize
        corner += corner'
        edge += ein"ijk -> kji"(conj.(edge))

        # normalize
        corner /= norm(corner)
        edge /= norm(edge)

        if norm(vals .- valsold) < tol
            break
        end
    end

    corner, edge
end

function energy(h, ipeps, χ; tol = 1e-10, maxit = 100)
    D = size(ipeps, 1) ^ 2
    s = size(ipeps, 5)
    ipeps = symmetrize(ipeps)
    ap = ein"abcdx, ijkly -> aibjckdlxy"(ipeps, conj.(ipeps))
    ap = reshape(ap, D, D, D, D, s, s)
    a = ein"ijklaa -> ijkl"(ap)

    rt = initialize(a, χ)
    expectationvalue(h, ap, ctmrg(a, rt, χ, D; tol = tol, maxit = maxit))
end

function expectationvalue(h, ap, rt)
    corner, edge = rt
    ap /= norm(ap)
    l = ein"ab, (ica, (bde, (cjfdlm, (eg, gfk)))) -> ijklm"(corner, edge, edge, ap, corner, edge) # fix later
    e = ein"(abcij, abckl), ijkl -> "(l, l, h)[]
    n = ein"ijkaa, ijkbb -> "(l, l)[]
    e / n
end

function vipeps(ipeps, h; χ = 20, tol = 1e-10, f_tol = 1e-8, maxit = 100)
    f(x) = real(energy(h, x, χ; tol = tol, maxit = maxit))
    res = optimize(f, f', ipeps, LBFGS(), Optim.Options(show_trace = true); inplace = false)
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
    vipeps(ipeps, real.(h ./ 2))
end

main()