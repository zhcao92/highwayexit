function relaxed_initial_state(mdp::NoCrashProblem, steps=200,
                             rng=MersenneTwister(rand(UInt32));
                             solver=BehaviorSolver(NORMAL, true, rng))

    pp = mdp.dmodel.phys_param
    is = MLState(0.0, 0.0, CarState[CarState(pp.lane_length/2, 4.0, pp.v_med, 0.0, NORMAL, 1)], true)
    sim = HistoryRecorder(max_steps=steps, rng=rng)
    policy = solve(solver, mdp)
    #policy = RandomPolicy(mdp)
    simulate_ini(sim, mdp, policy, is)
    #println("simulate done")
    return sim.state_hist[end]
end



##########################  change it

function relaxed_initial_state_cz(mdp::NoCrashProblem, is::MLState, policy::Policy, steps=20
                             #rng=MersenneTwister(rand(UInt32));
                             )
                             #solver=BehaviorSolver(NORMAL, true, rng))
                             #solver=SimpleSolver())

    #pp = mdp.dmodel.phys_param
    #is = MLState(0.0, 0.0, CarState[CarState(pp.lane_length/2, 1.0, pp.v_med, 0.0, NORMAL, 1)])
    rng = MersenneTwister(5)
    sim = HistoryRecorder(max_steps=steps, rng=rng)
    #policy = RandomPolicy(mdp)
    #policy = solve(solver, mdp)
    #policy = Simple(mdp)
    #print("sim")
    simulate_cz(sim, mdp, policy, is)
    #return sim.state_hist[end]

    #print("simulate done")
    return sim
end


function relaxed_initial_state_cz2(mdp::NoCrashProblem, is::MLState, steps=200,
                                 rng=MersenneTwister(rand(UInt32));
                                 solver=BehaviorSolver(NORMAL, true, rng))

    pp = mdp.dmodel.phys_param
    #is = MLState(0.0, 0.0, CarState[CarState(pp.lane_length/2, 4.0, pp.v_med, 0.0, NORMAL, 1)], true)
    sim = HistoryRecorder(max_steps=steps, rng=rng)
    policy = solve(solver, mdp)
    #policy = RandomPolicy(mdp)
    simulate_ini(sim, mdp, policy, is)
    #println("simulate done")
    return sim
end


###############################
#=
struct BehaviorSolver <: Solver
    b::BehaviorModel
    keep_lane::Bool
    rng::AbstractRNG
end

struct BehaviorPolicy <: Policy
    problem::NoCrashProblem
    b::BehaviorModel
    keep_lane::Bool
    rng::AbstractRNG
end
solve(s::BehaviorSolver, p::NoCrashProblem) = BehaviorPolicy(p, s.b, s.keep_lane, s.rng)
=#
############################

function simulate_cz{S,A}(sim::HistoryRecorder,
                       mdp::MDP{S,A}, policy::Policy
                      ,init_state::S=get_initial_state(sim, mdp)
                      )

                      #    init_state = get_initial_state(sim, mdp)

    max_steps = get(sim.max_steps, typemax(Int))
    if !isnull(sim.eps)
        max_steps = min(max_steps, ceil(Int,log(get(sim.eps))/log(discount(mdp))))
    end
    sizehint = get(sim.sizehint, min(max_steps, 1000))

    # aliases for the histories to make the code more concise
    sh = sim.state_hist = sizehint!(Vector{S}(0), sizehint)
    ah = sim.action_hist = sizehint!(Vector{A}(0), sizehint)
    oh = sim.observation_hist = Any[]
    bh = sim.belief_hist = Any[]
    rh = sim.reward_hist = sizehint!(Vector{Float64}(0), sizehint)

    if sim.show_progress
       prog = Progress(max_steps, "Simulating..." )
    end

    push!(sh, init_state)

    disc = 1.0
    step = 1
    try
        while !isterminal(mdp, sh[step]) && step <= max_steps
            action_calculate = action(policy, sh[step])
            action_use = control_action(mdp,sh[step])
            push!(ah, action(policy, sh[step]))
        #    push!(ah, action_use)
            sp, r = generate_sr(mdp, sh[step], ah[step], sim.rng)

        #    if action_calculate == action_use
        #       sp.Control_Signal = sh[step].Control_Signal
        #    else
        #       sp.Control_Signal = false
        #    end

            push!(sh, sp)
            push!(rh, r)

            disc *= discount(mdp)
            step += 1


            if sim.show_progress
                    next!(prog)
            end
        end
    catch ex
        if sim.capture_exception
            sim.exception = ex
            sim.backtrace = catch_backtrace()
        else
            rethrow(ex)
        end
    end

    if sim.show_progress
        finish!(prog)
    end

    return MDPHistory(sh, ah, rh, discount(mdp), sim.exception, sim.backtrace)

    return 0

end



function simulate_ini{S,A}(sim::HistoryRecorder,
                       mdp::MDP{S,A}, policy::Policy
                      ,init_state::S=get_initial_state(sim, mdp)
                      )

                      #    init_state = get_initial_state(sim, mdp)

    max_steps = get(sim.max_steps, typemax(Int))
    if !isnull(sim.eps)
        max_steps = min(max_steps, ceil(Int,log(get(sim.eps))/log(discount(mdp))))
    end
    sizehint = get(sim.sizehint, min(max_steps, 1000))

    # aliases for the histories to make the code more concise
    sh = sim.state_hist = sizehint!(Vector{S}(0), sizehint)
    ah = sim.action_hist = sizehint!(Vector{A}(0), sizehint)
    oh = sim.observation_hist = Any[]
    bh = sim.belief_hist = Any[]
    rh = sim.reward_hist = sizehint!(Vector{Float64}(0), sizehint)

    if sim.show_progress
       prog = Progress(max_steps, "Simulating..." )
    end

    push!(sh, init_state)

    disc = 1.0
    step = 1
    try
        while !isterminal(mdp, sh[step]) && step <= max_steps
            push!(ah, action(policy, sh[step]))
            sp, r = generate_sr(mdp, sh[step], ah[step], sim.rng)
            push!(sh, sp)
            push!(rh, r)

            disc *= discount(mdp)
            step += 1


            if sim.show_progress
                    next!(prog)
            end
        end
    catch ex
        if sim.capture_exception
            sim.exception = ex
            sim.backtrace = catch_backtrace()
        else
            rethrow(ex)
        end
    end

    if sim.show_progress
        finish!(prog)
    end

    return MDPHistory(sh, ah, rh, discount(mdp), sim.exception, sim.backtrace)

    return 0

end
