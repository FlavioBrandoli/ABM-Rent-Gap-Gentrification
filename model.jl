
function initialize_model(;
    dims = (40, 40), p_public_space_inner_city_non_cbd = 0.17, 
    p_public_space_suburbs = 0.20,
    cbd_radius_fraction = 0.08,
    max_house_age = 50.0, 
    age_stochasticity_std = 5.0,
    maintenance_noise_std = 0.05,
    amenity_distribution = Uniform(0.1, 1.0),
    base_potential_rent = 3, amenity_influence_radius = 2,
    amenity_rent_factor = 22, rent_age_decay = 0.02, seed = 12345,
    N_search_attempts = 10,
    decay_factor_age_slope = 1.5,
    satisfaction_threshold_low = 0.3, 
    satisfaction_threshold_high = 0.6, 
    min_rent_periphery_factor = 0.7,
    developer_search_sample_size = 30,
    developer_investments_per_step = 1,
    age_step = 0.05, 
    developer_evaluation_period = 2,
    developer_confidence_boost = 0.5, 
    developer_confidence_penalty = 0.5,
    developer_confidence_threshold = 0.4, 
    gentrification_spillover_radius = 1, 
    gentrification_spillover_boost = 4.0,
    plot_ranges = (
    rent = (0.0, 70.0),
    amenity = (0.0, 1.0), 
    age = (0.0, 70.0),
    potential_rent = (17.5, 70.0)
    ),
    developer_stopped_duration_threshold = 4, 
    developer_confidence_reset_value = 1.,
    developer_time_stopped = 0,
    mean_age_after_renovation = 7,

    public_housing_fraction = 0.05,
    public_housing_rent = 0.0,
    median_rent_percentage_threshold = 0.5,
    public_housing_negative_spillover_boost = 2.0,
    percentile_to_calculate = 0.10,
    public_housing_income_threshold = 30.0,
    public_housing_conversion_radius = 1,

    income_bin_edges_k = [0, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 75, 100, 125, 150, 200, 300],
    income_bin_weights = [8.38, 6.21, 6.24, 6.23, 5.93, 6.0, 5.41, 5.47, 4.85, 
    9.25, 11.5, 14.1, 9.29, 5.57, 5.65, 5.53],
    w_poor_percentile = 0.6,

    target_vacancy_rate = 0.075,
   

)
    rng = MersenneTwister(seed)
    space = GridSpace(dims, periodic = false)

    patch_type = fill(:empty, dims)
    house_age_matrix = fill(NaN, dims) 
    amenity_level = fill(NaN, dims)
    potential_rent = fill(NaN, dims)
    capitalized_rent = fill(NaN, dims)
    raw_amenity_scores_matrix = fill(NaN, dims)
    house_condition_noise = fill(0.0, dims)  
    management_type = fill(:empty_mgmt, dims)

    cbd_center = dims .÷ 2 .+ 1
    cbd_r = minimum(dims) * cbd_radius_fraction
    
    dx_max_dist = max(cbd_center[1]-1, dims[1]-cbd_center[1])
    dy_max_dist = max(cbd_center[2]-1, dims[2]-cbd_center[2])

    max_dist_from_center_to_corner = sqrt(dx_max_dist^2 + dy_max_dist^2)
    if max_dist_from_center_to_corner < 1e-9; max_dist_from_center_to_corner = 1.0; end


    max_dist_from_cbd_edge_for_rent_decay = max_dist_from_center_to_corner - cbd_r
    if max_dist_from_cbd_edge_for_rent_decay < 1e-9 
        max_dist_from_cbd_edge_for_rent_decay = 1.0 
    end

    inner_r_start = floor(Int, dims[1] / 4) + 1
    inner_r_end   = floor(Int, dims[1] * 3 / 4)
    inner_c_start = floor(Int, dims[2] / 4) + 1
    inner_c_end   = floor(Int, dims[2] * 3 / 4)

    total_initial_houses_count = 0
    num_public_housing_target = 0

    for j in 1:dims[2], i in 1:dims[1]
        pos = (i, j)
        dist_from_center = sqrt((i - cbd_center[1])^2 + (j - cbd_center[2])^2)

        if dist_from_center <= cbd_r
            patch_type[i, j] = :cbd
        else
            is_pos_in_defined_inner_city_block = (inner_r_start <= i <= inner_r_end) && 
                                                 (inner_c_start <= j <= inner_c_end)
            
            current_p_public_for_this_patch = is_pos_in_defined_inner_city_block ? p_public_space_inner_city_non_cbd : p_public_space_suburbs

            if rand(rng) < current_p_public_for_this_patch
                patch_type[i, j] = :public
                amenity_level[i, j] = rand(rng, amenity_distribution)
            else
                patch_type[i, j] = :house
                management_type[i,j] = :private
                total_initial_houses_count += 1

                dist_from_cbd_edge_for_age = max(0.0, dist_from_center - cbd_r)
                denominator_age_norm = (dx_max_dist+max_dist_from_center_to_corner)/2 - cbd_r 
                age_norm_factor = 0.0
                if denominator_age_norm > eps()
                    age_norm_factor = clamp(dist_from_cbd_edge_for_age / denominator_age_norm, 0.0, 1.0)
                end
                mean_age_at_location = max_house_age * (1.0 - age_norm_factor)   
                age_dist = Truncated(Normal(mean_age_at_location, age_stochasticity_std), 0.0, max_house_age) 
                current_stochastic_age = rand(rng, age_dist)
                house_age_matrix[i, j] = clamp(current_stochastic_age, 0.0, max_house_age) 
            
                noise_val = rand(rng, Normal(0, maintenance_noise_std))
                house_condition_noise[i, j] = noise_val
            
            end
        end
    end

    
    if total_initial_houses_count > 0
        N_agents = round(Int, total_initial_houses_count * (1.0 - target_vacancy_rate))
        #println("Case totali: $(total_initial_houses_count). Tasso di sfitto target: $(target_vacancy_rate*100)%. Numero agenti da creare: $(N_agents)")
    else
        N_agents = 0
        #println("Nessuna casa creata, nessun agente da inizializzare.")
    end
    
    num_public_housing_target = round(total_initial_houses_count*public_housing_fraction)

    for j in 1:dims[2], i in 1:dims[1]
        if patch_type[i, j] == :house
            pos = (i, j)
            neighbor_positions = Agents.nearby_positions(pos, space, amenity_influence_radius)
            sum_amenity = 0.0
            for np_tuple in neighbor_positions
                if patch_type[np_tuple...] == :public && !isnan(amenity_level[np_tuple...])
                    distance_from_house = maximum([abs(pos[1]-np_tuple[1]),abs(pos[2]-np_tuple[2])])
                    if distance_from_house > 0 
                        sum_amenity += amenity_level[np_tuple...] / (distance_from_house*1.5)
                    elseif distance_from_house == 0 && amenity_influence_radius == 0 
                        sum_amenity += amenity_level[np_tuple...] 
                    end
                end
            end
            raw_amenity_scores_matrix[i,j] = sum_amenity

            dist_from_center_for_rent = sqrt((i - cbd_center[1])^2 + (j - cbd_center[2])^2)
            dist_from_cbd_edge_for_rent = max(0.0, dist_from_center_for_rent - cbd_r)
            normalized_dist_for_rent_decay = clamp(dist_from_cbd_edge_for_rent / max_dist_from_cbd_edge_for_rent_decay, 0.0, 1.0)
            rent_distance_multiplier = 1.0 - normalized_dist_for_rent_decay * (1.0 - min_rent_periphery_factor)
            current_base_rent = base_potential_rent * rent_distance_multiplier
            pot_rent = current_base_rent + amenity_rent_factor * sum_amenity 
            potential_rent[i, j] = pot_rent
            
            age_val = house_age_matrix[i, j]
            decay_factor_age = max(0.0, 1.0 - (rent_age_decay * age_val)/decay_factor_age_slope) 

            noise_val_stored = house_condition_noise[i, j]
            final_condition_factor = clamp(decay_factor_age + noise_val_stored, 0.08, 1.0) 

            cap_rent = pot_rent * final_condition_factor
            capitalized_rent[i, j] = cap_rent
        end
    end

    all_valid_raw_scores = filter(x -> !isnan(x), vec(raw_amenity_scores_matrix))
    global_min_raw_score = isempty(all_valid_raw_scores) ? 0.0 : minimum(all_valid_raw_scores)
    global_max_raw_score = isempty(all_valid_raw_scores) ? 0.0 : maximum(all_valid_raw_scores)


    all_capitalized_rents = Float64[]
    for j in 1:dims[2], i in 1:dims[1]
        if patch_type[i, j] == :house && !isnan(capitalized_rent[i, j])
            push!(all_capitalized_rents, capitalized_rent[i, j])
        end
    end

    local calculated_public_housing_rent_val = 0.0
    local calculated_initial_median_inner_city_rent_val = 0.0 
    local calculated_percentile_to_calculate = 0.30
    
    inner_city_initial_rents = Float64[]
    for j in 1:dims[2], i in 1:dims[1]
        
        is_pos_in_defined_inner_city_block = (inner_r_start <= i <= inner_r_end) && (inner_c_start <= j <= inner_c_end)
        if patch_type[i, j] == :house && !isnan(capitalized_rent[i, j]) && is_pos_in_defined_inner_city_block
            push!(inner_city_initial_rents, capitalized_rent[i, j])
        end
    end

    if !isempty(all_capitalized_rents)
        if length(all_capitalized_rents) > 1
            calculated_public_housing_rent_val = quantile(all_capitalized_rents, calculated_percentile_to_calculate)
        else
            calculated_public_housing_rent_val = all_capitalized_rents[1]
        end
        #println("Affitto Public Housing calcolato (Percentile $(calculated_percentile_to_calculate*100)%): $(round(calculated_public_housing_rent_val, digits=2))")
    else
        #println("Nessuna casa con affitto capitalizzato trovata. Affitto Public Housing sarà 0.0.")
    end

    if !isempty(inner_city_initial_rents) 
        if length(inner_city_initial_rents) > 1
            calculated_initial_median_inner_city_rent_val = median(inner_city_initial_rents)
        else
            calculated_initial_median_inner_city_rent_val = inner_city_initial_rents[1]
        end
        #println("Mediana Affitto Iniziale (Inner City): $(round(calculated_initial_median_inner_city_rent_val, digits=2))")
    else
        #println("Nessuna casa nell'Inner City con affitto capitalizzato trovata. Mediana iniziale Inner City sarà 0.0.")
    end
    
    probabilities = income_bin_weights ./ sum(income_bin_weights)

    cumulative_prob = 0.0
    income_cutoff = 0.0
    
    for i in 1:length(probabilities)
        prob_in_this_bin = probabilities[i]
        
        if cumulative_prob + prob_in_this_bin >= w_poor_percentile
            
            prob_needed = w_poor_percentile - cumulative_prob
            
            fraction_of_bin = prob_needed / prob_in_this_bin
            
            lower_bound = income_bin_edges_k[i]
            upper_bound = income_bin_edges_k[i+1]
            bin_width = upper_bound - lower_bound
            
            income_cutoff = lower_bound + fraction_of_bin * bin_width
            break 
        end
        
        cumulative_prob += prob_in_this_bin
    end

    cumulative_prob2 = 0.0
    medium_high_income_cutoff = 0.0
    medium_high_percentile = 0.8
    
    for i in 1:length(probabilities)
        prob_in_this_bin = probabilities[i]
        
        if cumulative_prob2 + prob_in_this_bin >= medium_high_percentile

            prob_needed = medium_high_percentile - cumulative_prob2
            
            fraction_of_bin = prob_needed / prob_in_this_bin
            
            lower_bound = income_bin_edges_k[i]
            upper_bound = income_bin_edges_k[i+1]
            bin_width = upper_bound - lower_bound
            
            medium_high_income_cutoff = lower_bound + fraction_of_bin * bin_width
            break 
        end
        
        cumulative_prob2 += prob_in_this_bin
    end

    properties = Dict(
        :patch_type => patch_type, 
        :house_age => house_age_matrix, 
        :house_condition_noise => house_condition_noise,
        :amenity_level => amenity_level,
        :potential_rent => potential_rent, 
        :capitalized_rent => capitalized_rent,

        :management_type => management_type,

        :grid_dims => dims, 
        :max_house_age => max_house_age, 
        :current_max_observed_age => max_house_age,
        :cbd_center => cbd_center, 
        :cbd_radius => cbd_r,
        :amenity_rent_factor => amenity_rent_factor, 
        :rent_age_decay => rent_age_decay, 
        :neighborhood_radius => 1, 
        :utility_weights_low => (financial=0.35, amenity=0.25, age=0.25, center=0.15, neighborhood=0.0), 
        :utility_weights_high => (financial=0.25, amenity=0.21, age=0.36, center=0.18, neighborhood=0.0), 
        
        :step => 0, 
        :rng => rng, 
        :space => space, 
        :raw_amenity_scores_matrix => raw_amenity_scores_matrix,
        :global_min_raw_amenity_score => global_min_raw_score,
        :global_max_raw_amenity_score => global_max_raw_score, 
        :amenity_influence_radius => amenity_influence_radius,
        :N_search_attempts => N_search_attempts,
        :satisfaction_threshold_low => satisfaction_threshold_low,
        :satisfaction_threshold_high => satisfaction_threshold_high,
        :decay_factor_age_slope => decay_factor_age_slope, 
        :base_potential_rent => base_potential_rent,
        :min_rent_periphery_factor => min_rent_periphery_factor,
        :max_dist_from_center_to_corner => max_dist_from_center_to_corner,
        :p_public_space_inner_city_non_cbd => p_public_space_inner_city_non_cbd,
        :p_public_space_suburbs => p_public_space_suburbs,
        :developer_search_sample_size => developer_search_sample_size,
        :developer_investments_per_step => developer_investments_per_step,
        :age_step => age_step,
        :investments_to_evaluate => Dict{NTuple{2, Int}, Int}(), 
        :developer_confidence => 1.0,
        
        :developer_evaluation_period => developer_evaluation_period, 
        :developer_confidence_boost => developer_confidence_boost, 
        :developer_confidence_penalty => developer_confidence_penalty, 
        :developer_confidence_threshold => developer_confidence_threshold,
        :developer_activity_status => :active,
        :gentrification_spillover_radius => gentrification_spillover_radius,
        :gentrification_spillover_boost => gentrification_spillover_boost,
        :plot_rent_range => plot_ranges.rent,           
        :plot_amenity_range => plot_ranges.amenity,     
        :plot_age_range => plot_ranges.age,         
        :plot_potential_rent_range => plot_ranges.potential_rent,
        :developer_stopped_duration_threshold => developer_stopped_duration_threshold,
        :developer_confidence_reset_value => developer_confidence_reset_value,  
        :developer_time_stopped => developer_time_stopped, 
        :mean_age_after_renovation => mean_age_after_renovation, 
        :developer_recovery_threshold => 0.5, 
        :failed_investments => Set{NTuple{2, Int}}(), 

        
        :mean_quadrant_amenities_map => Dict{NTuple{2, Int}, Float64}(), 
        :quadrant_non_cbd_patch_counts => Dict{NTuple{2, Int}, Int}(),

        :public_housing_fraction => public_housing_fraction,
        :public_housing_rent => calculated_public_housing_rent_val,
        :public_housing_active => false, 
        :median_rent_percentage_threshold => median_rent_percentage_threshold, 
        :public_housing_negative_spillover_boost => public_housing_negative_spillover_boost,
        :initial_median_inner_city_rent => calculated_initial_median_inner_city_rent_val,
        :public_housing_income_threshold => public_housing_income_threshold,
        :public_housing_conversion_radius => public_housing_conversion_radius,
        :renovated_houses_history => Set{NTuple{2, Int}}(),
        :total_initial_houses_count => total_initial_houses_count,
        :num_public_housing_target => num_public_housing_target,


        :income_cutoff => income_cutoff,
        :medium_high_income_cutoff => medium_high_income_cutoff,

        :w_poor_percentile => w_poor_percentile,
        :real_income_midpoints => [5, 12.5, 17.5, 22.5, 27.5, 32.5, 37.5, 42.5, 47.5, 55, 67.5, 87.5, 112.5, 137.5, 175, 225],
        :real_income_weights => income_bin_weights,
        :real_income_bins => income_bin_edges_k,


        :renovated_houses_history_inner_city_steps => [],
        :renovated_houses_history_steps => [],
        :investments_this_step => 0
   
    )

    model = AgentBasedModel(AgentType, space; properties, rng, scheduler = my_custom_random_scheduler)

    model.mean_quadrant_amenities_map,
    model.quadrant_non_cbd_patch_counts = calculate_quadrant_amenity_scores(model)


    house_positions = findall(pt -> pt == :house, model.patch_type) 
    actual_N_agents = N_agents
    if N_agents > length(house_positions)
        #println("Attenzione: Numero agenti ($N_agents) > numero case ($(length(house_positions))). Riduco N_agents a $(length(house_positions)).")
        actual_N_agents = length(house_positions)
    end
    if isempty(house_positions) && actual_N_agents > 0
        #error("Nessuna casa generata sulla griglia, impossibile aggiungere agenti!")
    end
    shuffle!(model.rng, house_positions)
    
    for _ in 1:N_agents 
        
        if isempty(house_positions)
            #println("WARN: Finite le house_positions prima di creare tutti gli agenti.")
            break
        end
        current_house_pos_cartesian = pop!(house_positions)
        current_house_pos_tuple = Tuple(current_house_pos_cartesian) 

        income_val = sample_from_histogram(rng, income_bin_edges_k, probabilities)

        if income_val < income_cutoff
            add_agent!(current_house_pos_tuple, LowIncome, model, income_val)
        else
            add_agent!(current_house_pos_tuple, HighIncome, model, income_val)
        end
    end
    
    
    #println("Modello inizializzato con $actual_N_agents agenti.")
    num_low_income = count(a -> a isa LowIncome, allagents(model))
    num_high_income = count(a -> a isa HighIncome, allagents(model))
    #println("Creati $num_low_income LowIncome e $num_high_income HighIncome agenti.")
    #println("Tipi di patch: ", countmap(vec(model.patch_type)))
    return model
end








function agent_step!(agent::AgentType, model::ABM)
    current_pos = agent.pos
    current_utility = calculate_total_utility(agent, current_pos, model)

    local agent_specific_threshold::Float64
    if agent isa LowIncome
        agent_specific_threshold = model.satisfaction_threshold_low
    else 
        agent_specific_threshold = model.satisfaction_threshold_high
    end

    if current_utility < agent_specific_threshold
        
        local vacant_houses_pool_for_agent 

        if !model.public_housing_active 

            vacant_houses_pool_for_agent = get_vacant_house_positions(model) 
            
            if isempty(vacant_houses_pool_for_agent)
                return
            end

            num_to_sample = min(length(vacant_houses_pool_for_agent), model.N_search_attempts)
            
            houses_to_evaluate = if num_to_sample > 0
                
                custom_sample_without_replacement(model.rng, vacant_houses_pool_for_agent, num_to_sample)
            else
                NTuple{2, Int}[]
            end
            
        else 

            local vacant_private_houses_available = get_vacant_private_houses_positions(model)
            local vacant_public_housing_available = get_vacant_public_housing_positions(model)

            if agent isa LowIncome && agent.income_value < model.public_housing_income_threshold
               
                num_ph_to_consider = min(length(vacant_public_housing_available), model.N_search_attempts)
                
                if num_ph_to_consider > 0
                    vacant_houses_pool_for_agent = custom_sample_without_replacement(model.rng, vacant_public_housing_available, num_ph_to_consider)
                else
                    vacant_houses_pool_for_agent = NTuple{2, Int}[] 
                end

                remaining_attempts = model.N_search_attempts - length(vacant_houses_pool_for_agent)
                if remaining_attempts > 0 && !isempty(vacant_private_houses_available)
                    num_private_to_sample = min(length(vacant_private_houses_available), remaining_attempts)
                    private_sampled = custom_sample_without_replacement(model.rng, vacant_private_houses_available, num_private_to_sample)
                    append!(vacant_houses_pool_for_agent, private_sampled)
                end
                
            else 
                vacant_houses_pool_for_agent = custom_sample_without_replacement(model.rng, vacant_private_houses_available, min(length(vacant_private_houses_available), model.N_search_attempts))
                
            end

            if isempty(vacant_houses_pool_for_agent)
                return 
            end
            houses_to_evaluate = vacant_houses_pool_for_agent 
        end
       
        best_new_utility = -1.0 
        best_new_pos = nothing

        for potential_pos in houses_to_evaluate
            utility_at_potential_pos = calculate_total_utility(agent, potential_pos, model)
            if utility_at_potential_pos > best_new_utility
                best_new_utility = utility_at_potential_pos
                best_new_pos = potential_pos
            end
        end

        if best_new_pos !== nothing && best_new_utility > current_utility
            move_agent!(agent, best_new_pos, model)
        end
    end
end


function model_step_termalizzazione!(model::ABM) 
     push!(model.renovated_houses_history_steps, 0)
     push!(model.renovated_houses_history_inner_city_steps, 0)
end


function model_step_dinamica_free!(model::ABM)

    #println("\n--- Step Dinamico: $(model.step) ---")
    #println("Developer Status Iniziale: Fiducia = $(round(model.developer_confidence, digits=3)), Attività = $(model.developer_activity_status), Time Stopped = $(model.developer_time_stopped)")


    positions_evaluated_this_step = NTuple{2, Int}[] 

    for (pos, renovation_step) in pairs(model.investments_to_evaluate)
        occupants = ids_in_position(pos, model)
        if !isempty(occupants)
            model.developer_confidence += model.developer_confidence_boost
            push!(positions_evaluated_this_step, pos)
        else
            age_of_investment = model.step - renovation_step
            if age_of_investment >= model.developer_evaluation_period
                
                model.developer_confidence -= model.developer_confidence_penalty

                push!(model.failed_investments, pos)

                push!(positions_evaluated_this_step, pos)
            end
        end
    end

    for pos in positions_evaluated_this_step
        delete!(model.investments_to_evaluate, pos)
    end

    model.developer_confidence = clamp(model.developer_confidence, 0.0, 3)

    #println("Developer - Azione di Investimento:")
    
    renovated_this_step = NTuple{2, Int}[]
    vacant_houses_pos = get_vacant_house_positions(model)
    
    local num_to_renovate::Int = 0
    local rent_gaps = [] 

    if model.developer_activity_status == :stopped
        #println("  Developer è nello stato :stopped. Controllo le condizioni di ripresa...")
        
        num_failed = length(model.failed_investments)
        if num_failed > 0
            occupied_failures = 0
            
            for pos in model.failed_investments
                if !isempty(ids_in_position(pos, model))
                    occupied_failures += 1
                end
            end
            
            recovery_fraction = occupied_failures / num_failed
            #println("  - Valutazione investimenti passati: $occupied_failures / $num_failed ($(round(recovery_fraction * 100, digits=1))%) occupati.")

            if recovery_fraction >= model.developer_recovery_threshold
                #println("  [ATTIVITÀ RIPRESA] La frazione di recupero ($(round(recovery_fraction, digits=2))) ha superato la soglia ($(model.developer_recovery_threshold)).")
                model.developer_activity_status = :active
                model.developer_confidence = model.developer_confidence_reset_value 
                empty!(model.failed_investments)
            else
                #println("  - Il mercato non si è ancora ripreso a sufficienza. Il developer resta fermo.")
            end
        else
            #println("  - Nessun investimento fallito registrato. Il developer resta fermo.")
        end
    end

    if model.developer_confidence < model.developer_confidence_threshold && model.developer_activity_status == :active
        #println("  [ATTIVITÀ INTERROTTA] Fiducia ($(round(model.developer_confidence, digits=3))) < soglia ($(model.developer_confidence_threshold)).")
        model.developer_activity_status = :stopped
    end 
        
    if model.developer_activity_status == :active

        if !isempty(vacant_houses_pos)
            num_to_sample_dev = min(length(vacant_houses_pos), model.developer_search_sample_size)
            sampled_vacant_houses = StatsBase.sample(model.rng, vacant_houses_pos, num_to_sample_dev, replace=false)

            for pos_tuple in sampled_vacant_houses
                pr = model.potential_rent[pos_tuple...]
                cr = model.capitalized_rent[pos_tuple...]
                if !isnan(pr) && !isnan(cr) && pr > cr
                    push!(rent_gaps, (gap = pr - cr, position = pos_tuple))
                end
            end

            if !isempty(rent_gaps)
                sort!(rent_gaps, by = x -> x.gap, rev=true)
                
                max_investments = model.developer_investments_per_step
                desired_investments = max_investments * model.developer_confidence
                num_to_renovate = round(Int, desired_investments)
                num_to_renovate = min(num_to_renovate, length(rent_gaps))
            else
                #println("  - Nessuna casa con 'rent gap' positivo trovata nel campione, nonostante fiducia sufficiente.")
                num_to_renovate = 0 
            end
        else
            #println("  - Nessuna casa sfitta disponibile in città per investimenti.")
            num_to_renovate = 0 
        end
        #println("  - Investimenti PREVISTI (Stato: $(model.developer_activity_status), Fiducia: $(round(model.developer_confidence, digits=3))): $num_to_renovate")
    end 

    #println("  - Investimenti ESEGUITI questo step: $num_to_renovate")

    push!(model.renovated_houses_history_steps, num_to_renovate)
    
    counter = 0
    
    if num_to_renovate > 0 && !isempty(rent_gaps)
        for i in 1:num_to_renovate
            house_to_renovate_pos_tuple = rent_gaps[i].position
           
            if is_in_inner_city(house_to_renovate_pos_tuple, model)
                counter+=1
            end

            effective_new_age = rand(model.rng, Exponential(model.mean_age_after_renovation))
            model.house_age[house_to_renovate_pos_tuple...] = effective_new_age
            model.investments_to_evaluate[house_to_renovate_pos_tuple] = model.step
            
            push!(renovated_this_step, house_to_renovate_pos_tuple)
        
        end
    end

    push!(model.renovated_houses_history_inner_city_steps, counter)

    if !isempty(renovated_this_step)
        neighbors_to_boost = Set{NTuple{2, Int}}()
        for pos_renovated in renovated_this_step
            neighbor_positions = nearby_positions(pos_renovated, model, model.gentrification_spillover_radius)
            for neighbor_pos in neighbor_positions
                if model.patch_type[neighbor_pos...] == :house && !(neighbor_pos in renovated_this_step)
                    push!(neighbors_to_boost, neighbor_pos)
                end
            end
        end
        if !isempty(neighbors_to_boost) 
            #println("  - Effetto Spillover: $(length(neighbors_to_boost)) case vicine vedono un aumento del Potential Rent.")
            for pos_to_boost in neighbors_to_boost
                model.potential_rent[pos_to_boost...] += model.gentrification_spillover_boost
            end
        end
    end

    age_increment = model.age_step
    rent_age_decay_param = model.rent_age_decay
    decay_factor_age_slope_param = model.decay_factor_age_slope

    for pos_cartesian in findall(p -> p == :house, model.patch_type)
        current_house_age = model.house_age[pos_cartesian]
        if isnan(current_house_age); continue; end

        base_condition_factor = max(0.0, 1.0 - (rent_age_decay_param * current_house_age) / decay_factor_age_slope_param)
        noise_val = model.house_condition_noise[pos_cartesian]
        final_condition_factor = clamp(base_condition_factor + noise_val, 0.1, 1.0)
        current_pr = model.potential_rent[pos_cartesian]
        new_cr = current_pr * final_condition_factor
        model.capitalized_rent[pos_cartesian] = clamp(new_cr, 0.0, current_pr)
        model.house_age[pos_cartesian] = current_house_age + age_increment
    end
    
    all_house_ages_in_step = filter(age_val -> !isnan(age_val), vec(model.house_age))
    if !isempty(all_house_ages_in_step)
        model.current_max_observed_age = maximum(all_house_ages_in_step)
    end
    
    model.current_max_observed_age = max(model.current_max_observed_age, model.max_house_age) 

    model.step += 1
    #println("----------------------------------------")
    return nothing
end


function model_step_dinamica_public_random!(model::ABM)

    #println("\n--- Step Dinamico: $(model.step) ---")
    flush(stdout) 
    #println("Developer Status Iniziale: Fiducia = $(round(model.developer_confidence, digits=3)), Attività = $(model.developer_activity_status), Time Stopped = $(model.developer_time_stopped)")
    flush(stdout) 
    
    current_inner_city_rent_metrics = calculate_inner_city_rent_metric(model)
    local current_median_inner_city_rent = current_inner_city_rent_metrics.median_inner_city_rent 

    
    local rent_growth_percentage = 0.0
    
    if model.initial_median_inner_city_rent > 0 
        rent_growth_percentage = (current_median_inner_city_rent - model.initial_median_inner_city_rent) / model.initial_median_inner_city_rent
    end
    
    #println("  Mediana Affitto Inner City: $(round(current_median_inner_city_rent, digits=2)), Crescita: $(round(rent_growth_percentage*100, digits=1))% (Soglia: $(round(model.median_rent_percentage_threshold*100, digits=1))%)")
    flush(stdout)

    if !model.public_housing_active
        
        if rent_growth_percentage >= model.median_rent_percentage_threshold
            model.public_housing_active = true
            #println("!!! POLITICA DI PUBLIC HOUSING ATTIVATA allo Step: $(model.step) (Crescita Affitto: $(round(rent_growth_percentage*100, digits=1))% >= Soglia: $(round(model.median_rent_percentage_threshold*100, digits=1))%) !!!")
            flush(stdout)

            
            perform_public_housing_conversion!(model)
        end
    end


    positions_evaluated_this_step = NTuple{2, Int}[]

    for (pos, renovation_step) in pairs(model.investments_to_evaluate)
        occupants = ids_in_position(pos, model)
        if !isempty(occupants)
            model.developer_confidence += model.developer_confidence_boost
            push!(positions_evaluated_this_step, pos)
        else
            age_of_investment = model.step - renovation_step
            if age_of_investment >= model.developer_evaluation_period
                model.developer_confidence -= model.developer_confidence_penalty
                push!(model.failed_investments, pos)
                push!(positions_evaluated_this_step, pos)
            end
        end
    end

    for pos in positions_evaluated_this_step
        delete!(model.investments_to_evaluate, pos)
    end

    model.developer_confidence = clamp(model.developer_confidence, 0.0, 3)


    #println("Developer - Azione di Investimento:")
    
    renovated_this_step = NTuple{2, Int}[]
    local vacant_houses_pos_for_developer 

    if !model.public_housing_active 
        
        vacant_houses_pos_for_developer = get_vacant_house_positions(model) 
        #println("  Developer cerca casa con logica BASELINE.")
        flush(stdout)
    else 
        vacant_houses_pos_for_developer = get_vacant_private_houses_positions(model)
        #println("  Developer cerca casa con logica PUBLIC HOUSING (solo private).")
        flush(stdout)
    end
    
    local num_to_renovate::Int = 0
    local rent_gaps = [] 

    if model.developer_activity_status == :stopped
        #println("  Developer è nello stato :stopped. Controllo le condizioni di ripresa...")
        flush(stdout)
        num_failed = length(model.failed_investments)
        if num_failed > 0
            occupied_failures = 0
            for pos in model.failed_investments
                if !isempty(ids_in_position(pos, model))
                    occupied_failures += 1
                end
            end
            recovery_fraction = occupied_failures / num_failed
            #println("  - Valutazione investimenti passati: $occupied_failures / $num_failed ($(round(recovery_fraction * 100, digits=1))%) occupati.")
            flush(stdout)
            if recovery_fraction >= model.developer_recovery_threshold
                #println("  [ATTIVITÀ RIPRESA] La frazione di recupero ($(round(recovery_fraction, digits=2))) ha superato la soglia ($(model.developer_recovery_threshold)).")
                flush(stdout)
                model.developer_activity_status = :active
                model.developer_confidence = model.developer_confidence_reset_value 
                empty!(model.failed_investments) 
            else
                #println("  - Il mercato non si è ancora ripreso a sufficienza. Il developer resta fermo.")
                flush(stdout)
            end
        else
            #println("  - Nessun investimento fallito registrato. Il developer resta fermo.")
            flush(stdout)
        end
    end

    if model.developer_confidence < model.developer_confidence_threshold && model.developer_activity_status == :active
        #println("  [ATTIVITÀ INTERROTTA] Fiducia ($(round(model.developer_confidence, digits=3))) < soglia ($(model.developer_confidence_threshold)).")
        flush(stdout)
        model.developer_activity_status = :stopped
    end
    
    if model.developer_activity_status == :active
        if !isempty(vacant_houses_pos_for_developer)
            num_to_sample_dev = min(length(vacant_houses_pos_for_developer), model.developer_search_sample_size)
            sampled_vacant_houses = StatsBase.sample(model.rng, vacant_houses_pos_for_developer, num_to_sample_dev, replace=false)

            for pos_tuple in sampled_vacant_houses
                pr = model.potential_rent[pos_tuple...]
                cr = model.capitalized_rent[pos_tuple...]
                if !isnan(pr) && !isnan(cr) && pr > cr
                    push!(rent_gaps, (gap = pr - cr, position = pos_tuple))
                end
            end

            if !isempty(rent_gaps)
                sort!(rent_gaps, by = x -> x.gap, rev=true)
                
                max_investments = model.developer_investments_per_step
                desired_investments = max_investments * model.developer_confidence
                num_to_renovate = round(Int, desired_investments)
                num_to_renovate = min(num_to_renovate, length(rent_gaps))
            else
                #println("  - Nessuna casa con 'rent gap' positivo trovata nel campione, nonostante fiducia sufficiente.")
                flush(stdout)
                num_to_renovate = 0
            end
        else
            #println("  - Nessuna casa sfitta disponibile in città per investimenti.")
            flush(stdout)
            num_to_renovate = 0
        end
        #println("  - Investimenti PREVISTI (Stato: $(model.developer_activity_status), Fiducia: $(round(model.developer_confidence, digits=3))): $num_to_renovate")
        flush(stdout)
    end
    #println("  - Investimenti ESEGUITI questo step: $num_to_renovate")
    flush(stdout)
    
    push!(model.renovated_houses_history_steps, num_to_renovate)
    
    counter = 0
    
    if num_to_renovate > 0 && !isempty(rent_gaps)
        for i in 1:num_to_renovate
            house_to_renovate_pos_tuple = rent_gaps[i].position
           
            if is_in_inner_city(house_to_renovate_pos_tuple, model)
                counter+=1
            end

            effective_new_age = rand(model.rng, Exponential(model.mean_age_after_renovation))
            model.house_age[house_to_renovate_pos_tuple...] = effective_new_age
            model.investments_to_evaluate[house_to_renovate_pos_tuple] = model.step
            
            push!(renovated_this_step, house_to_renovate_pos_tuple)
        
        end
    end

    push!(model.renovated_houses_history_inner_city_steps, counter)

    
    if !isempty(renovated_this_step)
        neighbors_to_boost = Set{NTuple{2, Int}}()
        for pos_renovated in renovated_this_step
            neighbor_positions = nearby_positions(pos_renovated, model, model.gentrification_spillover_radius)
            for neighbor_pos in neighbor_positions
                
                if !model.public_housing_active 
                    if model.patch_type[neighbor_pos...] == :house && !(neighbor_pos in renovated_this_step)
                        push!(neighbors_to_boost, neighbor_pos)
                    end
                else 
                    if model.patch_type[neighbor_pos...] == :house && model.management_type[neighbor_pos...] == :private && !(neighbor_pos in renovated_this_step)
                        push!(neighbors_to_boost, neighbor_pos)
                    end
                end
            end
        end
        if !isempty(neighbors_to_boost)
            #println("  - Effetto Spillover Positivo: $(length(neighbors_to_boost)) case vicine vedono un aumento del Potential Rent.")
            flush(stdout)
            for pos_to_boost in neighbors_to_boost
                model.potential_rent[pos_to_boost...] += model.gentrification_spillover_boost
            end
        end
    end

    if model.public_housing_active
        
        public_housing_patches = NTuple{2, Int}[] 
        for x_idx in 1:model.grid_dims[1], y_idx in 1:model.grid_dims[2]
            if model.management_type[x_idx, y_idx] == :public
                push!(public_housing_patches, (x_idx, y_idx))
            end
        end
        
        if !isempty(public_housing_patches)
            neighbors_to_penalize = Set{NTuple{2, Int}}()
            for pos_ph_house in public_housing_patches
                neighbor_positions = nearby_positions(pos_ph_house, model, model.public_housing_conversion_radius)
                for neighbor_pos in neighbor_positions
                   
                    if model.patch_type[neighbor_pos...] == :house && model.management_type[neighbor_pos...] == :private && !(neighbor_pos in model.renovated_houses_history)
                        push!(neighbors_to_penalize, neighbor_pos)
                    end
                end
            end
            if !isempty(neighbors_to_penalize)
                #println("  - Effetto Spillover Negativo da Public Housing: $(length(neighbors_to_penalize)) case vicine vedono una diminuzione del Potential Rent.")
                flush(stdout)
                for pos_to_penalize in neighbors_to_penalize
                    model.potential_rent[pos_to_penalize...] -= model.public_housing_negative_spillover_boost
                    model.potential_rent[pos_to_penalize...] = max(model.potential_rent[pos_to_penalize...], model.base_potential_rent * 0.5)
                end
            end
        end
    end

    age_increment = model.age_step
    rent_age_decay_param = model.rent_age_decay
    decay_factor_age_slope_param = model.decay_factor_age_slope

    for r_idx in 1:model.grid_dims[1], c_idx in 1:model.grid_dims[2]
        current_pos_tuple = (r_idx, c_idx) 

        if model.patch_type[r_idx, c_idx] == :house 
           
            if !model.public_housing_active 
                current_house_age = model.house_age[r_idx, c_idx] 
                if isnan(current_house_age); continue; end
                base_condition_factor = max(0.0, 1.0 - (rent_age_decay_param * current_house_age) / decay_factor_age_slope_param)
                noise_val = model.house_condition_noise[r_idx, c_idx]
                final_condition_factor = clamp(base_condition_factor + noise_val, 0.1, 1.0)
                current_pr = model.potential_rent[r_idx, c_idx]
                new_cr = current_pr * final_condition_factor
                model.capitalized_rent[r_idx, c_idx] = clamp(new_cr, 0.0, current_pr)
                model.house_age[r_idx, c_idx] = current_house_age + age_increment
            else 
                if model.management_type[r_idx, c_idx] == :public 
                    model.capitalized_rent[r_idx, c_idx] = model.public_housing_rent
                    
                else 
                    current_house_age = model.house_age[r_idx, c_idx]
                    if isnan(current_house_age); continue; end
                    base_condition_factor = max(0.0, 1.0 - (rent_age_decay_param * current_house_age) / decay_factor_age_slope_param)
                    noise_val = model.house_condition_noise[r_idx, c_idx]
                    final_condition_factor = clamp(base_condition_factor + noise_val, 0.1, 1.0)
                    current_pr = model.potential_rent[r_idx, c_idx]
                    new_cr = current_pr * final_condition_factor
                    model.capitalized_rent[r_idx, c_idx] = clamp(new_cr, 0.0, current_pr)
                    model.house_age[r_idx, c_idx] = current_house_age + age_increment
                end
            end
        end
    end 

    all_house_ages_in_step = filter(age_val -> !isnan(age_val), vec(model.house_age))
    if !isempty(all_house_ages_in_step)
        model.current_max_observed_age = maximum(all_house_ages_in_step)
    end
    model.current_max_observed_age = max(model.current_max_observed_age, model.max_house_age)

    model.step += 1
    #println("----------------------------------------")
    flush(stdout)
    return nothing
end

function model_step_dinamica_public_rent_gap!(model::ABM)

    #println("\n--- Step Dinamico: $(model.step) ---")
    flush(stdout) 
    #println("Developer Status Iniziale: Fiducia = $(round(model.developer_confidence, digits=3)), Attività = $(model.developer_activity_status), Time Stopped = $(model.developer_time_stopped)")
    flush(stdout) 

    current_inner_city_rent_metrics = calculate_inner_city_rent_metric(model)
    local current_median_inner_city_rent = current_inner_city_rent_metrics.median_inner_city_rent 

    
    local rent_growth_percentage = 0.0
    if model.initial_median_inner_city_rent > 0 
        rent_growth_percentage = (current_median_inner_city_rent - model.initial_median_inner_city_rent) / model.initial_median_inner_city_rent
    end
    
    #println("  Mediana Affitto Inner City: $(round(current_median_inner_city_rent, digits=2)), Crescita: $(round(rent_growth_percentage*100, digits=1))% (Soglia: $(round(model.median_rent_percentage_threshold*100, digits=1))%)")
    flush(stdout)

    if !model.public_housing_active
        
        if rent_growth_percentage >= model.median_rent_percentage_threshold
            model.public_housing_active = true
            #println("!!! POLITICA DI PUBLIC HOUSING ATTIVATA allo Step: $(model.step) (Crescita Affitto: $(round(rent_growth_percentage*100, digits=1))% >= Soglia: $(round(model.median_rent_percentage_threshold*100, digits=1))%) !!!")
            flush(stdout)

            
            perform_public_housing_conversion_by_rent_gap!(model)
        end
    end


    positions_evaluated_this_step = NTuple{2, Int}[]

    for (pos, renovation_step) in pairs(model.investments_to_evaluate)
        occupants = ids_in_position(pos, model)
        if !isempty(occupants)
            model.developer_confidence += model.developer_confidence_boost
            push!(positions_evaluated_this_step, pos)
        else
            age_of_investment = model.step - renovation_step
            if age_of_investment >= model.developer_evaluation_period
                model.developer_confidence -= model.developer_confidence_penalty
                push!(model.failed_investments, pos)
                push!(positions_evaluated_this_step, pos)
            end
        end
    end

    for pos in positions_evaluated_this_step
        delete!(model.investments_to_evaluate, pos)
    end

    model.developer_confidence = clamp(model.developer_confidence, 0.0, 3)


    #println("Developer - Azione di Investimento:")

    renovated_this_step = NTuple{2, Int}[]

    local vacant_houses_pos_for_developer 

    if !model.public_housing_active 
        vacant_houses_pos_for_developer = get_vacant_house_positions(model)
        #println("  Developer cerca casa con logica BASELINE.")
        flush(stdout)
    else 
        vacant_houses_pos_for_developer = get_vacant_private_houses_positions(model)
        #println("  Developer cerca casa con logica PUBLIC HOUSING (solo private).")
        flush(stdout)
    end
    
    local num_to_renovate::Int = 0
    local rent_gaps = []

    if model.developer_activity_status == :stopped
        #println("  Developer è nello stato :stopped. Controllo le condizioni di ripresa...")
        flush(stdout)
        num_failed = length(model.failed_investments)
        if num_failed > 0
            occupied_failures = 0
            for pos in model.failed_investments
                if !isempty(ids_in_position(pos, model))
                    occupied_failures += 1
                end
            end
            recovery_fraction = occupied_failures / num_failed
            #println("  - Valutazione investimenti passati: $occupied_failures / $num_failed ($(round(recovery_fraction * 100, digits=1))%) occupati.")
            flush(stdout)
            if recovery_fraction >= model.developer_recovery_threshold
                #println("  [ATTIVITÀ RIPRESA] La frazione di recupero ($(round(recovery_fraction, digits=2))) ha superato la soglia ($(model.developer_recovery_threshold)).")
                flush(stdout)
                model.developer_activity_status = :active
                model.developer_confidence = model.developer_confidence_reset_value 
                empty!(model.failed_investments)
            else
                #println("  - Il mercato non si è ancora ripreso a sufficienza. Il developer resta fermo.")
                flush(stdout)
            end
        else
            #println("  - Nessun investimento fallito registrato. Il developer resta fermo.")
            flush(stdout)
        end
    end

    if model.developer_confidence < model.developer_confidence_threshold && model.developer_activity_status == :active
        #println("  [ATTIVITÀ INTERROTTA] Fiducia ($(round(model.developer_confidence, digits=3))) < soglia ($(model.developer_confidence_threshold)).")
        flush(stdout)
        model.developer_activity_status = :stopped
    end
    
    if model.developer_activity_status == :active
        if !isempty(vacant_houses_pos_for_developer)
            num_to_sample_dev = min(length(vacant_houses_pos_for_developer), model.developer_search_sample_size)
            sampled_vacant_houses = StatsBase.sample(model.rng, vacant_houses_pos_for_developer, num_to_sample_dev, replace=false)

            for pos_tuple in sampled_vacant_houses
                pr = model.potential_rent[pos_tuple...]
                cr = model.capitalized_rent[pos_tuple...]
                if !isnan(pr) && !isnan(cr) && pr > cr
                    push!(rent_gaps, (gap = pr - cr, position = pos_tuple))
                end
            end

            if !isempty(rent_gaps)
                sort!(rent_gaps, by = x -> x.gap, rev=true)
                
                max_investments = model.developer_investments_per_step
                desired_investments = max_investments * model.developer_confidence
                num_to_renovate = round(Int, desired_investments)
                num_to_renovate = min(num_to_renovate, length(rent_gaps))
            else
                #println("  - Nessuna casa con 'rent gap' positivo trovata nel campione, nonostante fiducia sufficiente.")
                flush(stdout)
                num_to_renovate = 0
            end
        else
            #println("  - Nessuna casa sfitta disponibile in città per investimenti.")
            flush(stdout)
            num_to_renovate = 0
        end
        #println("  - Investimenti PREVISTI (Stato: $(model.developer_activity_status), Fiducia: $(round(model.developer_confidence, digits=3))): $num_to_renovate")
        flush(stdout)
    end
    #println("  - Investimenti ESEGUITI questo step: $num_to_renovate")
    flush(stdout)

    push!(model.renovated_houses_history_steps, num_to_renovate)
    
    counter = 0
    
    if num_to_renovate > 0 && !isempty(rent_gaps)
        for i in 1:num_to_renovate
            house_to_renovate_pos_tuple = rent_gaps[i].position
           
            if is_in_inner_city(house_to_renovate_pos_tuple, model)
                counter+=1
            end

            effective_new_age = rand(model.rng, Exponential(model.mean_age_after_renovation))
            model.house_age[house_to_renovate_pos_tuple...] = effective_new_age
            model.investments_to_evaluate[house_to_renovate_pos_tuple] = model.step
            
            push!(renovated_this_step, house_to_renovate_pos_tuple)
        
        end
    end

    push!(model.renovated_houses_history_inner_city_steps, counter)

    if !isempty(renovated_this_step)
        neighbors_to_boost = Set{NTuple{2, Int}}()
        for pos_renovated in renovated_this_step
            neighbor_positions = nearby_positions(pos_renovated, model, model.gentrification_spillover_radius)
            for neighbor_pos in neighbor_positions
                
                if !model.public_housing_active
                    
                    if model.patch_type[neighbor_pos...] == :house && !(neighbor_pos in renovated_this_step)
                        push!(neighbors_to_boost, neighbor_pos)
                    end
                else 
                    if model.patch_type[neighbor_pos...] == :house && model.management_type[neighbor_pos...] == :private && !(neighbor_pos in renovated_this_step)
                        push!(neighbors_to_boost, neighbor_pos)
                    end
                end
            end
        end
        if !isempty(neighbors_to_boost)
            #println("  - Effetto Spillover Positivo: $(length(neighbors_to_boost)) case vicine vedono un aumento del Potential Rent.")
            flush(stdout)
            for pos_to_boost in neighbors_to_boost
                model.potential_rent[pos_to_boost...] += model.gentrification_spillover_boost
            end
        end
    end

    if model.public_housing_active
      
        public_housing_patches = NTuple{2, Int}[] 
        for x_idx in 1:model.grid_dims[1], y_idx in 1:model.grid_dims[2]
            if model.management_type[x_idx, y_idx] == :public 
                push!(public_housing_patches, (x_idx, y_idx))
            end
        end

        if !isempty(public_housing_patches)
            neighbors_to_penalize = Set{NTuple{2, Int}}()
            for pos_ph_house in public_housing_patches
                neighbor_positions = nearby_positions(pos_ph_house, model, model.public_housing_conversion_radius)
                for neighbor_pos in neighbor_positions
                    
                    if model.patch_type[neighbor_pos...] == :house && model.management_type[neighbor_pos...] == :private && !(neighbor_pos in model.renovated_houses_history)
                        push!(neighbors_to_penalize, neighbor_pos)
                    end
                end
            end
            if !isempty(neighbors_to_penalize)
                #println("  - Effetto Spillover Negativo da Public Housing: $(length(neighbors_to_penalize)) case vicine vedono una diminuzione del Potential Rent.")
                flush(stdout)
                for pos_to_penalize in neighbors_to_penalize
                    model.potential_rent[pos_to_penalize...] -= model.public_housing_negative_spillover_boost
                    model.potential_rent[pos_to_penalize...] = max(model.potential_rent[pos_to_penalize...], model.base_potential_rent * 0.5)
                end
            end
        end
    end
    
    age_increment = model.age_step
    rent_age_decay_param = model.rent_age_decay
    decay_factor_age_slope_param = model.decay_factor_age_slope

    
    for r_idx in 1:model.grid_dims[1], c_idx in 1:model.grid_dims[2]
        current_pos_tuple = (r_idx, c_idx) 
        
        if model.patch_type[r_idx, c_idx] == :house 
            
            if !model.public_housing_active 
                current_house_age = model.house_age[r_idx, c_idx] 
                if isnan(current_house_age); continue; end
                base_condition_factor = max(0.0, 1.0 - (rent_age_decay_param * current_house_age) / decay_factor_age_slope_param)
                noise_val = model.house_condition_noise[r_idx, c_idx]
                final_condition_factor = clamp(base_condition_factor + noise_val, 0.1, 1.0)
                current_pr = model.potential_rent[r_idx, c_idx]
                new_cr = current_pr * final_condition_factor
                model.capitalized_rent[r_idx, c_idx] = clamp(new_cr, 0.0, current_pr)
                model.house_age[r_idx, c_idx] = current_house_age + age_increment
            else 
                if model.management_type[r_idx, c_idx] == :public 
                    model.capitalized_rent[r_idx, c_idx] = model.public_housing_rent
                    
                else 
                    current_house_age = model.house_age[r_idx, c_idx]
                    if isnan(current_house_age); continue; end
                    base_condition_factor = max(0.0, 1.0 - (rent_age_decay_param * current_house_age) / decay_factor_age_slope_param)
                    noise_val = model.house_condition_noise[r_idx, c_idx]
                    final_condition_factor = clamp(base_condition_factor + noise_val, 0.1, 1.0)
                    current_pr = model.potential_rent[r_idx, c_idx]
                    new_cr = current_pr * final_condition_factor
                    model.capitalized_rent[r_idx, c_idx] = clamp(new_cr, 0.0, current_pr)
                    model.house_age[r_idx, c_idx] = current_house_age + age_increment
                end
            end
        end
    end 

    all_house_ages_in_step = filter(age_val -> !isnan(age_val), vec(model.house_age))
    if !isempty(all_house_ages_in_step)
        model.current_max_observed_age = maximum(all_house_ages_in_step)
    end
    model.current_max_observed_age = max(model.current_max_observed_age, model.max_house_age)

    model.step += 1
    #println("----------------------------------------")
    flush(stdout)
    return nothing
end