function relaxed_initial_state_TRI(mdp::NoCrashProblem, LastState::MLState, MKZAction::MLAction, DSRC_interval::Float64
                             ,rng=MersenneTwister(rand(UInt32)))

    pp = mdp.dmodel.phys_param
    #sim = HistoryRecorder(max_steps=steps, rng=rng)
    #policy = solve(solver, mdp)
    #policy = RandomPolicy(mdp)
    CurrentState = simulate_TRI(mdp, MKZAction,LastState, DSRC_interval,rng)
    #println("simulate done")
    return CurrentState
end

function simulate_TRI{S,A}(mdp::MDP{S,A}, MKZAction::MLAction,LastState::MLState, DSRC_interval::Float64,rng::AbstractRNG)
    sp = generate_sTRI(mdp, LastState, MKZAction, DSRC_interval,rng)
    return sp
end


function generate_sTRI(mdp::NoCrashProblem, s::MLState, a::MLAction, DSRC_interval::Float64, rng::AbstractRNG)

    @if_debug dbg_rng = copy(rng)

    sp::MLState=create_state(mdp)
    pp = mdp.dmodel.phys_param
    dt = pp.dt
    nb_cars = length(s.cars)
    resize!(sp.cars, nb_cars)
    #sp.x = 0
    sp.terminal = s.terminal

    ## Calculate deltas ##
    #====================#



    dxs = Array{Float64}(nb_cars)
    dvs = Array{Float64}(nb_cars)
    dys = Array{Float64}(nb_cars)
    lcs = Array{Float64}(nb_cars)

    # agent
    dvs[1] = a.acc*dt
    dxs[1] = s.cars[1].vel*dt + a.acc*dt^2/2.
    lcs[1] = a.lane_change
    dys[1] = a.lane_change*dt

    for i in 2:nb_cars
        neighborhood = get_neighborhood(pp, s, i)

        behavior = s.cars[i].behavior

        acc = gen_accel(behavior, mdp.dmodel, s, neighborhood, i, rng)
        dvs[i] = dt*acc
        dxs[i] = (s.cars[i].vel + dvs[i]/2.)*dt

        lcs[i] = gen_lane_change(behavior, mdp.dmodel, s, neighborhood, i, rng)
        dys[i] = lcs[i] * dt
    end

    ## Consistency checking ##
    #========================#

    # first prevent lane changes into each other
    changers = IntSet()
    for i in 1:nb_cars
        if lcs[i] != 0
            push!(changers, i)
        end
    end
    sorted_changers = sort!(collect(changers), by=i->s.cars[i].x, rev=true) # this might be slow because anonymous functions are slow
    # from front to back

    if length(sorted_changers) >= 2 #something to compare
        # iterate through pairs
        iter_state = start(sorted_changers)
        j, iter_state = next(sorted_changers, iter_state)
        while !done(sorted_changers, iter_state)
            i = j
            j, iter_state = next(sorted_changers, iter_state)
            car_i = s.cars[i] # front
            car_j = s.cars[j] # back

            # check if they are both starting to change lanes on this step
            if isinteger(car_i.y) && isinteger(car_j.y)

                # check if they are near each other lanewise
                if abs(car_i.y - car_j.y) <= 2.0

                    # make sure there is a conflict longitudinally
                    # if car_i.x - car_j.x <= pp.l_car || car_i.x + dxs[i] - car_j.x + dxs[j] <= pp.l_car
                    # if car_i.x - car_j.x <= mdp.dmodel.appear_clearance # made more conservative on 8/19
                    # if car_i.x - car_j.x <= get_idm_s_star(car_j.behavior.p_idm, car_j.vel, car_j.vel-car_i.vel) # upgraded to sstar on 8/19
                    ivp = car_i.vel + dt*dvs[i]
                    jvp = car_j.vel + dt*dvs[j]
                    ixp = car_i.x + dt*(car_i.vel + ivp)/2.0
                    jxp = car_j.x + dt*(car_j.vel + jvp)/2.0
                    n_max_acc_p = nullable_max_safe_acc(ixp-jxp-pp.l_car, jvp, ivp, pp.brake_limit,dt)
                    if ixp - jxp <= pp.l_car || car_i.x - car_j.x <= pp.l_car || isnull(n_max_acc_p) || get(n_max_acc_p) < -pp.brake_limit

                        # check if they are moving towards each other
                        # if dys[i]*dys[j] < 0.0 && abs(car_i.y+dys[i] - car_j.y+dys[j]) < 2.0
                        if true # prevent lockstepping 8/19 (doesn't prevent it that well)

                            # make j stay in his lane
                            dys[j] = 0.0
                            lcs[j] = 0.0
                        end
                    end
                end
            end
        end
    end

    # second, prevent cars hitting each other due to noise
    sorted = sort!(collect(1:length(s.cars)), by=i->s.cars[i].x, rev=true)

    if length(sorted) >= 2 #something to compare
        # iterate through pairs
        iter_state = start(sorted)
        j, iter_state = next(sorted, iter_state)
        while !done(sorted, iter_state)
            i = j
            j, iter_state = next(sorted, iter_state)
            if j == 1
                continue # don't check for the ego since the ego does not have noise
            end
            car_i = s.cars[i]
            car_j = s.cars[j]

            # check if they overlap longitudinally
            if car_j.x + dxs[j] > car_i.x + dxs[i] - pp.l_car

                # check if they will be in the same lane
                if occupation_overlap(car_i.y + dys[i], car_j.y + dys[j])
                    # warn and nudge behind
                    @if_debug begin
                        println("Conflict because of noise: front:$i, back:$j")
                        Gallium.@enter generate_s(mdp, s, a, dbg_rng)
                    end
                    if i == 1
                         warn("Car nudged because noise would cause a crash (ego in front).")
                        #error("Car nudged because noise would cause a crash (ego in front).")
                    else
                         warn("Car nudged because noise would cause a crash.")
                        #error("Car nudged because noise would cause a crash.")
                    end
                    dxs[j] = car_i.x + dxs[i] - car_j.x - 1.01*pp.l_car
                    dvs[j] = 2.0*(dxs[j]/dt - car_j.vel)
                end
            end
        end
    end

    ## Dynamics and Exits ##
    #======================#

    exits = IntSet()
    for i in 1:nb_cars
        car = s.cars[i]
        xp = car.x + (dxs[i] - dxs[1])
        yp = car.y + dys[i]
        # velp = max(min(car.vel + dvs[i],pp.v_max), pp.v_min)
        velp = max(car.vel + dvs[i], 0.0) # removed speed limits on 8/13
        # note lane change is updated above

        if dvs[i]/dt < -mdp.dmodel.brake_terminate_thresh
            sp.terminal = Nullable{Symbol}(:brake)
        end

        # check if a lane was crossed and snap back to it
        if isinteger(car.y)
            # prevent a multi-lane change in a single timestep
            if abs(yp-car.y) > 1.
                yp = car.y + sign(dys[i])
            end
        else # car.y is not an integer
            if floor(yp) >= ceil(car.y)
                yp = ceil(car.y)
            end
            if ceil(yp) <= floor(car.y)
                yp = floor(car.y)
            end
        end

        # if yp < 1.0 || yp > pp.nb_lanes
        #     @show i
        #     @show yp
        #     println("mdp = $mdp")
        #     println("s = $s")
        #     println("a = $a")
        # end
        #print("yp")
        #println(yp)
        @assert yp >= 1.0 && yp <= pp.nb_lanes

        if xp < 0.0 || xp >= pp.lane_length
            push!(exits, i)
        else
            sp.cars[i] = CarState(xp, yp, velp, lcs[i], car.behavior, s.cars[i].id)
        end
    end

    next_id = maximum([c.id for c in s.cars]) + 1

    deleteat!(sp.cars, exits)
    nb_cars -= length(exits)

    ## Generate new cars ##
    #=====================#

    if nb_cars < mdp.dmodel.nb_cars && rand(rng) <= mdp.dmodel.p_appear

        behavior = rand(rng, mdp.dmodel.behaviors)
        vel = typical_velocity(behavior) + randn(rng)*mdp.dmodel.vel_sigma

        clearances = Array{Float64}(pp.nb_lanes)
        fill!(clearances, Inf)
        closest_cars = Array{Int}(pp.nb_lanes)
        fill!(closest_cars, 0)
        sstar_margins = Array{Float64}(pp.nb_lanes)
        if vel > s.cars[1].vel
            # put at back
            # sstar is the sstar of the new guy
            for i in 1:length(s.cars)
                lowlane, highlane = occupation_lanes(s.cars[i].y, lcs[i])
                back = s.cars[i].x - pp.l_car
                if back < clearances[lowlane]
                    clearances[lowlane] = back
                    closest_cars[lowlane] = i
                end
                if back < clearances[highlane]
                    clearances[highlane] = back
                    closest_cars[highlane] = i
                end
            end
            for j in 1:pp.nb_lanes
                other = closest_cars[j]
                if other == 0
                    sstar = 0
                else
                    sstar = get_idm_s_star(behavior.p_idm, vel, vel-s.cars[other].vel)
                end
                sstar_margins[j] = clearances[j] - sstar
            end
        else
            for i in 1:length(s.cars)
                lowlane, highlane = occupation_lanes(s.cars[i].y, lcs[i])
                front = pp.lane_length - (s.cars[i].x + pp.l_car) # l_car is half the length of the old car plus half the length of the new one
                if front < clearances[lowlane]
                    clearances[lowlane] = front
                    closest_cars[lowlane] = i
                end
                if front < clearances[highlane]
                    clearances[highlane] = front
                    closest_cars[highlane] = i
                end
            end
            for j in 1:pp.nb_lanes
                other = closest_cars[j]
                if other == 0
                    sstar = 0
                else
                    sstar = get_idm_s_star(s.cars[other].behavior.p_idm,
                                           s.cars[other].vel,
                                           s.cars[other].vel-vel)
                end
                sstar_margins[j] = clearances[j] - sstar
            end
        end

        margin, lane = findmax(sstar_margins)

        if margin > 0.0
            if vel > s.cars[1].vel
                # at back
                push!(sp.cars, CarState(0.0, lane, vel, 0.0, behavior, next_id))
            else
                push!(sp.cars, CarState(pp.lane_length, lane, vel, 0.0, behavior, next_id))
            end
        end
    end

    if mdp.dmodel.lane_terminate && sp.cars[1].y == mdp.rmodel.target_lane
        sp.terminal = Nullable{Any}(:lane)
    elseif sp.x > mdp.dmodel.max_dist
        sp.terminal = Nullable{Any}(:distance)
    end

    # sp.crashed = is_crash(mdp, s, sp, warning=false)
    # sp.crashed = false

    @assert sp.cars[1].x == s.cars[1].x # ego should not move
    sp.x = s.x + sp.cars[1].vel*0.75
    sp.t = s.t
    ca = control_action(mdp,s)
    if s.Control_Signal && (a == ca)
       sp.Control_Signal = true
    else
       sp.Control_Signal = false
    end

    if s.x > 2500 && isinteger(s.cars[1].y) && s.t == 0.0
        sp.Control_Signal = true
        sp.t = 1.0
    end

    #if s.cars[1].y == 4.0 && s.t == 0.0
    #    sp.Control_Signal = true
    #    sp.t = 1.0
    #end
    return sp
end
