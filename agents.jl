
Base.@kwdef mutable struct LowIncome <: AbstractAgent
    id::Int
    pos::NTuple{2, Int}
    income_value::Float64
end

Base.@kwdef mutable struct HighIncome <: AbstractAgent
    id::Int
    pos::NTuple{2, Int}
    income_value::Float64
end

AgentType = Union{LowIncome, HighIncome}