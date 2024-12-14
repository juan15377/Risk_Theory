using CairoMakie

# Libraries
using DataFrames
using LinearAlgebra

# Define the `Portafolio` structure
struct Portafolio
    insured_matrix::Matrix{Int64} # Matrix of insured values
    probabilities::Vector{Float64} # Vector of probabilities

    # Constructor for the Portafolio structure
    function Portafolio(insured_matrix, probabilities)
        I, J = size(insured_matrix)
        return new(insured_matrix, probabilities)
    end
end

# Define the probabilities and insured matrix
probabilidades = [0.03, 0.04, 0.05] # Probabilities for each risk category
matriz_de_asegurados = [
    1 3 1;
    2 3 4;
    3 5 4;
    4 2 6;
    5 2 4
]


portafolio = Portafolio(matriz_de_asegurados, probabilidades)

# Function `h` calculates auxiliary values for the portfolio
@inline function h(p::Portafolio, i::Int64, k::Int64)
    probabilities = p.probabilities
    insured_matrix = p.insured_matrix
    sum_ = 0
    for j in 1:size(insured_matrix, 2)
        sum_ += insured_matrix[i, j] * ((probabilities[j] / (1 - probabilities[j]))^k)
    end
    return (i * (-1)^(k - 1)) * sum_
end

# Recursive version of `pril_1` to calculate probabilities
function pril_1(p::Portafolio, x::Int64)
    insured_matrix = p.insured_matrix
    probabilities = p.probabilities

    I, J = size(insured_matrix)

    # Calculate the maximum possible sum
    s_max = sum((1:I) .* sum(insured_matrix, dims=2))

    # Return 0 if x exceeds the maximum sum
    if x > s_max
        return 0
    end

    # Base case: probability at x = 0
    g0 = 1
    @inbounds for i in 1:I
        for j in 1:J
            g0 *= ((1 - probabilities[j])^insured_matrix[i, j])
        end
    end
    if x == 0
        return g0
    end

    # Recursive calculation for probabilities
    sum_ = BigFloat(0)
    for i in 1:min(x, I)
        for k in 1:floor(x / i)
            sum_ += pril_1(p, Int64(x - i * k)) * h(p, i, Int64(k))
        end
    end
    return 1 / x * sum_
end

# Memoized version of `pril_1` to calculate all probabilities
function pril_1(p::Portafolio)
    insured_matrix = p.insured_matrix
    probabilities = p.probabilities

    I, J = size(insured_matrix)
    proba = Dict{Int64, Float64}() # Dictionary to store probabilities
    s_max = sum((1:I) .* sum(insured_matrix, dims=2))

    # Calculate the base probability for x = 0
    g0 = 1
    @inbounds for i in 1:I
        for j in 1:J
            g0 *= ((1 - probabilities[j])^insured_matrix[i, j])
        end
    end
    proba[0] = g0

    # Iterate to calculate probabilities for all x
    x = 1
    @inbounds while x <= s_max
        suma = BigFloat(0)
        for i in 1:min(x, I)
            for k in 1:floor(x / i)
                suma += proba[x - i * k] * h(p, i, Int64(k))
            end
        end
        proba[x] = (1 / x) * suma
        x += 1
    end
    return proba
end


# Calculate probabilities with memoization
@time probas = pril_1(portafolio)

# Verify that probabilities sum to 1
@assert sum(i[2] for i in probas) ≈ 1.0

# Plot the first 20 probabilities
eje_x = [i for i in 0:20]
eje_y = [probas[i] for i in 0:20]

# Change the plot theme
theme(:dracula)

# Create a bar plot
fig = Figure(size = (900, 600))
ax = Axis(fig[1,1], title = "Density Function", ylabel = "proba", xlabel = "x")
barplot!(ax, eje_x, eje_y, color = "blue", label = "Probabilities", alpha = 1)
fig 

save("pril_1.png", fig)

# Alternative recursive probability calculation
function pril_2(n::Int, f::Array)
    if n == 0
        return (f[1])^n
    end

    s = BigFloat(0)
    for j in 1:n
        if j + 1 > length(f)
            break
        end
        s += (((j * (n + 1)) / n) - 1) * f[j + 1] * pril_2(n - j, f)
    end
    s *= (1 / f[1])
    return s
end

# Example usage of pril_2
pril_2(1, [0.5, 0.3, 0.2])
