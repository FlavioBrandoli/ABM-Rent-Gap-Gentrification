function my_custom_random_scheduler(model::ABM)
    agent_ids = collect(allids(model)) # Prende tutti gli ID e li mette in un vettore
    return shuffle(model.rng, agent_ids) # Mescola gli ID usando il generatore casuale del modello
end


function sample_from_histogram(rng, bin_edges::Vector, probabilities::Vector; min_value = 7.0)
    
    @assert length(bin_edges) == length(probabilities) + 1 "Deve esserci un bordo in più rispetto alle probabilità."
    
    
    bin_index = wsample(rng, 1:length(probabilities), probabilities)
    
    
    lower_bound = bin_edges[bin_index]
    upper_bound = bin_edges[bin_index + 1]
    
    
    if bin_index == 1
        lower_bound = max(lower_bound, min_value)
    end
    
    return rand(rng) * (upper_bound - lower_bound) + lower_bound
end


function calculate_financial_utility(agent::AgentType, house_pos::NTuple{2, Int}, model::ABM)
    
    income = agent.income_value
    rent_matrix = model.capitalized_rent
    rent = rent_matrix[house_pos...]

    if income <= 0.0
        return rent > 0.0 ? 0.0 : 1.0 
    end

    if rent <= 0.0
        return 1.0
    end

    if rent >= income
        return 0.0
    end

    financial_utility = (income - rent) / income
    
    return clamp(financial_utility, 0.0, 1.0)
end


function calculate_amenity_utility(house_pos::NTuple{2, Int}, model::ABM)
    
    current_raw_score = model.raw_amenity_scores_matrix[house_pos...]

    if isnan(current_raw_score)
        
        println("Attenzione: punteggio amenità grezzo NaN per la posizione $house_pos")
        return 0.0
    end

    overall_min_score = model.global_min_raw_amenity_score
    overall_max_score = model.global_max_raw_amenity_score

    
    local normalized_amenity_utility::Float64
    range_amenity = overall_max_score - overall_min_score

    if abs(range_amenity) < 1e-9 
        normalized_amenity_utility = (abs(overall_min_score) < 1e-9 && abs(current_raw_score) < 1e-9) ? 0.0 : 0.5
    else
        normalized_amenity_utility = (current_raw_score - overall_min_score) / range_amenity
    end
    
    
    normalized_amenity_utility = clamp(normalized_amenity_utility, 0.0, 1.0)

    return normalized_amenity_utility
end


function calculate_neighborhood_utility(agent::AgentType, house_pos::NTuple{2, Int}, model::ABM)
    # Trova tutti gli agenti nel raggio specificato
    neighbor_ids = nearby_ids(house_pos, model, model.neighborhood_radius)
    
    # Conta il numero totale di vicini
    num_neighbors = count(i -> true, neighbor_ids)
    
    # Se non ci sono vicini, l'utilità di questa componente è neutra o nulla. Restituiamo 0.
    if num_neighbors == 0
        return 0.0
    end
    
    # Conta quanti vicini sono dello stesso tipo dell'agente che sta valutando la casa
    same_type_neighbors = 0
    for id in neighbor_ids
        neighbor_agent = model[id]
        if typeof(neighbor_agent) == typeof(agent)
            same_type_neighbors += 1
        end
    end
    
    # L'utilità è la frazione di vicini dello stesso tipo
    return same_type_neighbors / num_neighbors
end


function calculate_total_utility(agent::AgentType, house_pos::NTuple{2, Int}, model::ABM)
    
    rent_val = model.capitalized_rent[house_pos...]
    if isnan(rent_val)
        
        return 0.0
    end

    u_fin = calculate_financial_utility(agent, house_pos, model)

    if u_fin == 0.0 && rent_val > 0.0
        return 0.0
    end

    age_matrix = model.house_age
    u_amenity = calculate_amenity_utility(house_pos, model)

    
    current_house_age = age_matrix[house_pos...] 
    u_age_calculated = 0.0 
    if !isnan(current_house_age) 
        if model.current_max_observed_age > 0.0 
            raw_u_age = (model.current_max_observed_age - current_house_age) / model.current_max_observed_age
            u_age_calculated = clamp(raw_u_age, 0.0, 1.0)
        elseif model.current_max_observed_age == 0.0 && current_house_age == 0.0 
            
            u_age_calculated = 1.0 
        end
    end

    if u_age_calculated < 0.05
        return 0.0
    end
    


    dist_from_center = sqrt((house_pos[1] - model.cbd_center[1])^2 + (house_pos[2] - model.cbd_center[2])^2)
    u_center = clamp(1.0 - dist_from_center / model.max_dist_from_center_to_corner, 0.0, 1.0)

    
    u_neighborhood = calculate_neighborhood_utility(agent, house_pos, model)

    local weights
    if agent isa LowIncome
        weights = model.utility_weights_low 
    else 
        weights = model.utility_weights_high
    end

    
    total_u_raw = (weights.financial * u_fin +
                   weights.amenity * u_amenity +
                   weights.age * u_age_calculated +
                   weights.center * u_center +
                   weights.neighborhood * u_neighborhood) 
                      
    return clamp(total_u_raw, 0.0, 1.0)


end


function agent_colour_abm(a)

    return a isa LowIncome ? :blue : :red

end



function get_low_income_color_from_norm(normalized_income::Float64)
    # Sfumatura di Blu più acceso:
    # Min (norm=0): Blu (0.0, 0.2, 0.7)
    # Max (norm=1): Azzurro/Ciano brillante (0.2, 0.7, 1.0)
    r_val = 0.0 + 0.2 * normalized_income
    g_val = 0.2 + 0.5 * normalized_income
    b_val = 0.7 + 0.3 * normalized_income
    return RGBA(r_val, g_val, b_val, 1.0)
end

function get_high_income_color_from_norm(normalized_income::Float64)
    # Sfumatura di Rosso più "puro" / magenta, per distinguerlo dall'arancione:
    # Min (norm=0): Rosso rubino/violaceo scuro (0.7, 0.0, 0.3)
    # Max (norm=1): Rosso-Magenta brillante (1.0, 0.3, 0.6)
    r_val = 0.7 + 0.3 * normalized_income
    g_val = 0.0 + 0.3 * normalized_income # Ridotta componente verde
    b_val = 0.3 + 0.3 * normalized_income # Aumentata componente blu per spingere verso magenta
    
    
    return RGBA(r_val, g_val, b_val, 1.0)
end

function calculate_quadrant_amenity_scores(model::ABM)
    # Dizionario per accumulare la SOMMA dei punteggi di amenity delle aree pubbliche per ogni quadrante
    quadrant_public_amenity_sum = Dict{NTuple{2, Int}, Float64}()
    # Dizionario per contare il numero di patches NON-CBD (public + house) in ogni quadrante
    quadrant_non_cbd_patch_count = Dict{NTuple{2, Int}, Int}()

    grid_dims = model.grid_dims # Es: (40, 40)
    quadrant_size_x = div(grid_dims[1], 4) # Es: 10
    quadrant_size_y = div(grid_dims[2], 4) # Es: 10

    for x in 1:grid_dims[1]
        for y in 1:grid_dims[2]
            current_patch_type = model.patch_type[x, y] # Usa gli indici corretti x, y

            # Salta le patch CBD
            if current_patch_type == :cbd
                continue
            end

            # Calcola il quadrante per la cella (x,y)
            qx = ceil(Int, x / quadrant_size_x)
            qy = ceil(Int, y / quadrant_size_y)
            quadrant_coords = (qx, qy)

            # Incrementa il contatore delle patch non-CBD per il quadrante corrente
            quadrant_non_cbd_patch_count[quadrant_coords] = get(quadrant_non_cbd_patch_count, quadrant_coords, 0) + 1

            if current_patch_type == :public
                # Recupera l'amenity score per questa cella pubblica
                amenity_score = model.amenity_level[x, y]

                # Accumula i punteggi di amenity solo per le aree pubbliche
                quadrant_public_amenity_sum[quadrant_coords] = get(quadrant_public_amenity_sum, quadrant_coords, 0.0) + amenity_score
            end
            # Le patch :house non aggiungono amenity direttamente alla somma,
            # ma contribuiscono al denominatore (quadrant_non_cbd_patch_count)
        end
    end

    # Calcola la media delle amenity per ogni quadrante secondo la tua nuova definizione:
    # Somma delle amenity delle aree pubbliche / Numero totale di patch non-CBD (public + house)
    mean_quadrant_amenities = Dict{NTuple{2, Int}, Float64}()
    
    # Itera su tutti i quadranti che contengono patch non-CBD
    for (quadrant_coords, total_non_cbd_patches) in quadrant_non_cbd_patch_count
        sum_amenity_in_quadrant = get(quadrant_public_amenity_sum, quadrant_coords, 0.0)
        
        # Evita la divisione per zero se per qualche motivo un quadrante non ha patch non-CBD
        # (anche se con la tua griglia 40x40 e 16 quadranti, è improbabile)
        if total_non_cbd_patches > 0
            mean_quadrant_amenities[quadrant_coords] = sum_amenity_in_quadrant / total_non_cbd_patches
        else
            mean_quadrant_amenities[quadrant_coords] = 0.0 # O NaN, a seconda di come vuoi gestire i quadranti vuoti
        end
    end
    
    # Restituisce sia la media delle amenity per quadrante, sia il conteggio delle patch non-CBD
    return mean_quadrant_amenities, quadrant_non_cbd_patch_count
end


function city_plot(model::ABM)
    # --- CALCOLI PRELIMINARI PER LE SFUMATURE (Agenti) ---
    # Questa parte per il reddito degli agenti rimane invariata, perché il loro colore
    # può tranquillamente essere relativo allo step corrente.
    all_incomes_list = [a.income_value for a in allagents(model) if hasproperty(a, :income_value) && isa(a.income_value, Real)]
    min_inc_actual = isempty(all_incomes_list) ? 0.0 : minimum(all_incomes_list)
    max_inc_actual = isempty(all_incomes_list) ? 1.0 : (min_inc_actual == maximum(all_incomes_list) ? min_inc_actual + 1.0 : maximum(all_incomes_list))
    
    ## MODIFICA 1: Rimuovi i calcoli dinamici per i limiti di amenità e età.
    # all_amenities_values = filter(x -> !isnan(x), vec(model.amenity_level)) 
    # min_amenity = ...
    # max_amenity = ...
    #
    # min_house_age_for_plot = 0.0
    # max_house_age_for_plot = model.current_max_observed_age 
    # if max_house_age_for_plot <= min_house_age_for_plot ...
    
    ## MODIFICA 2: Recupera i limiti FISSI dalle proprietà del modello.
    # Assicurati che :plot_age_range e :plot_amenity_range siano stati aggiunti
    # alle properties del modello nella funzione initialize_model.
    min_age_plot, max_age_plot = model.plot_age_range
    min_amenity_plot, max_amenity_plot = model.plot_amenity_range

    # --- FUNZIONE COLORE AGENTI (invariata) ---
    function agent_color_by_income(agent)
        if !hasproperty(agent, :income_value) || !isa(agent.income_value, Real)
            return RGBA(0.5, 0.5, 0.5, 1.0) 
        end
        range_inc = max_inc_actual - min_inc_actual
        normalized_income = range_inc > 1e-9 ? (agent.income_value - min_inc_actual) / range_inc : 0.5
        normalized_income = clamp(normalized_income, 0.0, 1.0)
        
        if agent isa LowIncome; return get_low_income_color_from_norm(normalized_income);
        elseif agent isa HighIncome; return get_high_income_color_from_norm(normalized_income);
        else; return RGBA(0.0, 0.0, 0.0, 1.0); end
    end

    # --- COSTRUZIONE DELLA COLORMAP DISCRETA PER LE PATCH (invariata) ---
    N_house_age_bins = 20
    N_amenity_bins = 20  
    final_colormap = RGBA{Float64}[]
    # ... (il resto della costruzione della colormap non cambia) ...
    id_empty = 1
    push!(final_colormap, RGBA(1.0, 1.0, 1.0, 1.0)) 

    base_r_yellow = 1.0; base_g_yellow = 0.95; base_b_yellow = 0.7; 
    base_r_orange = 0.9; base_g_orange = 0.45; base_b_orange = 0.0; 
    house_age_sub_colormap = RGBA{Float64}[]
    for i_bin in 0:(N_house_age_bins-1)
        norm_val_for_bin = N_house_age_bins == 1 ? 0.5 : i_bin / (N_house_age_bins - 1)
        r_val = base_r_yellow + norm_val_for_bin * (base_r_orange - base_r_yellow)
        g_val = base_g_yellow + norm_val_for_bin * (base_g_orange - base_g_yellow)
        b_val = base_b_yellow + norm_val_for_bin * (base_b_orange - base_b_yellow)
        push!(final_colormap, RGBA(r_val, g_val, b_val, 1.0))
        push!(house_age_sub_colormap, RGBA(r_val, g_val, b_val, 1.0))
    end
    offset_after_house = id_empty + N_house_age_bins

    base_r_dark_green = 0.1; base_g_dark_green = 0.3; base_b_dark_green = 0.1;
    base_r_light_green = 0.6; base_g_light_green = 1.0; base_b_light_green = 0.6;
    amenity_greens_sub_colormap = RGBA{Float64}[] 
    for i_bin in 0:(N_amenity_bins-1)
        norm_val_for_bin = N_amenity_bins == 1 ? 0.5 : i_bin / (N_amenity_bins - 1)
        r_val = base_r_dark_green + norm_val_for_bin * (base_r_light_green - base_r_dark_green)
        g_val = base_g_dark_green + norm_val_for_bin * (base_g_light_green - base_g_dark_green)
        b_val = base_b_dark_green + norm_val_for_bin * (base_b_light_green - base_b_dark_green)
        push!(final_colormap, RGBA(r_val, g_val, b_val, 1.0))
        push!(amenity_greens_sub_colormap, RGBA(r_val, g_val, b_val, 1.0))
    end
    offset_after_public = offset_after_house + N_amenity_bins

    id_cbd = offset_after_public + 1
    push!(final_colormap, RGBA(0.2, 0.2, 0.2, 1.0)) 
    total_colormap_entries = length(final_colormap)

    # --- CREAZIONE MATRICE DEGLI ID PER LA COLORMAP ---
    patch_ids_for_colormap = zeros(Int, size(model.patch_type))
    try
        for i_cartesian_idx in CartesianIndices(model.patch_type)
            patch_sym = model.patch_type[i_cartesian_idx]
            if patch_sym == :house
                house_age_val = model.house_age[i_cartesian_idx]
                
                ## MODIFICA 3: Normalizza usando i limiti fissi per l'età.
                range_age_plot = max_age_plot - min_age_plot
                normalized_age = 0.0
                if !isnan(house_age_val) && range_age_plot > 1e-9
                    normalized_age = (house_age_val - min_age_plot) / range_age_plot
                end
                normalized_age = clamp(normalized_age, 0.0, 1.0) # Clamp per sicurezza
                
                bin_index_age = floor(Int, normalized_age * (N_house_age_bins - 1e-9))
                bin_index_age = clamp(bin_index_age, 0, N_house_age_bins - 1)
                patch_ids_for_colormap[i_cartesian_idx] = id_empty + 1 + bin_index_age 
            
            elseif patch_sym == :public
                amenity_val = model.amenity_level[i_cartesian_idx]

                ## MODIFICA 4: Normalizza usando i limiti fissi per le amenità.
                range_amenity_plot = max_amenity_plot - min_amenity_plot
                normalized_amenity = 0.0
                if !isnan(amenity_val) && range_amenity_plot > 1e-9
                    normalized_amenity = (amenity_val - min_amenity_plot) / range_amenity_plot
                end
                normalized_amenity = clamp(normalized_amenity, 0.0, 1.0) # Clamp per sicurezza

                bin_index_amenity = floor(Int, normalized_amenity * (N_amenity_bins - 1e-9))
                bin_index_amenity = clamp(bin_index_amenity, 0, N_amenity_bins - 1)
                patch_ids_for_colormap[i_cartesian_idx] = offset_after_house + 1 + bin_index_amenity
            
            elseif patch_sym == :empty
                patch_ids_for_colormap[i_cartesian_idx] = id_empty
            elseif patch_sym == :cbd
                patch_ids_for_colormap[i_cartesian_idx] = id_cbd
            else 
                patch_ids_for_colormap[i_cartesian_idx] = id_empty 
            end
        end
    catch e; println("ERRORE durante creazione patch_ids_for_colormap:");showerror(stdout, e);println(); end
    
    if isempty(patch_ids_for_colormap); println("ERRORE CRITICO: patch_ids_for_colormap è vuota."); return; end

    # --- PLOTTING (invariato) ---
    local fig_abm, ax_abm
    plot_successful = false
    try
        fig_abm, ax_abm, _ = abmplot( 
            model;
            agent_color = agent_color_by_income, agent_size = 12, agent_marker = :circle,
            heatarray = _ -> patch_ids_for_colormap,
            heatkwargs = ( colormap = final_colormap, colorrange = (1, total_colormap_entries) ),
            figure = (; resolution = (1200, 900)), 
            #axis = (; ),
            add_colorbar = false 
        )
        plot_successful = true 
    catch e_plot; println("ERRORE durante abmplot: "); showerror(stdout, e_plot); println(); return; end
    
    # --- Disegno rettangolo (invariato) ---
    if plot_successful && isdefined(Main, :get_inner_city_bounds) 
        try
            r_start, r_end, c_start, c_end = get_inner_city_bounds(model)
            x_coords = [c_start - 0.5, c_end + 0.5, c_end + 0.5, c_start - 0.5, c_start - 0.5]
            y_coords = [r_start - 0.5, r_start - 0.5, r_end + 0.5, r_end + 0.5, r_start - 0.5]
            Makie.lines!(ax_abm, x_coords, y_coords, color = :black, linewidth = 2, linestyle = :dash)
        catch e_rect; println("ERRORE disegno rettangolo Inner City: "); showerror(stdout, e_rect); println(); end
    end

    if plot_successful
        ## MODIFICA 5: Usa i limiti fissi nella creazione delle colorbar.
        
        # Colorbar per Amenity
        Makie.Colorbar(fig_abm[1, 2], colormap = amenity_greens_sub_colormap, limits = (min_amenity_plot, max_amenity_plot), label = "Amenity Level", width = 30,labelsize = 30)
        
        # Colorbar per House Age/Deterioration
        Makie.Colorbar(fig_abm[1, 3], colormap = house_age_sub_colormap, limits = (min_age_plot, max_age_plot), label = "House Age", width = 30,labelsize = 30)
        
        # --- Il resto della funzione per le colorbar degli agenti e il layout è invariato ---
        num_cb_samples = 50 
        if isdefined(Main, :get_low_income_color_from_norm)
            low_income_cb_colors = [get_low_income_color_from_norm(n) for n in range(0, 1, length=num_cb_samples)]
            Makie.Colorbar(fig_abm[2, 1], colormap = low_income_cb_colors, limits = (min_inc_actual, max_inc_actual), label = "Low-income", vertical = false, height = 30, flipaxis = false,labelsize = 30 )
        end
        if isdefined(Main, :get_high_income_color_from_norm)
            high_income_cb_colors = [get_high_income_color_from_norm(n) for n in range(0, 1, length=num_cb_samples)]
            Makie.Colorbar(fig_abm[3, 1], colormap = high_income_cb_colors, limits = (min_inc_actual, max_inc_actual), label = "High-income", vertical = false, height = 30, flipaxis = false,labelsize = 30 )
        end
        
       #= Makie.colsize!(fig_abm.layout, 1, Makie.Relative(0.75))
        Makie.colsize!(fig_abm.layout, 2, Makie.Relative(0.125))
        Makie.colsize!(fig_abm.layout, 3, Makie.Relative(0.125))
        Makie.rowsize!(fig_abm.layout, 1, Makie.Relative(0.80))
        Makie.rowsize!(fig_abm.layout, 2, Makie.Relative(0.10))
        Makie.rowsize!(fig_abm.layout, 3, Makie.Relative(0.10))=#

        colsize!(fig_abm.layout, 1, Aspect(1, 1.0))

        Makie.colgap!(fig_abm.layout, 1, 30)
        Makie.colgap!(fig_abm.layout, 2, 30)
        Makie.rowgap!(fig_abm.layout, 1, 30)
        Makie.rowgap!(fig_abm.layout, 2, 30)

        display(fig_abm)
        println("Visualizzazione ABM con colorbar finali generata.")
    end
end



function city_plot_rent(model::ABM)
    # --- CALCOLI PRELIMINARI PER LE SFUMATURE (Agenti, invariato) ---
    all_incomes = [a.income_value for a in allagents(model) if isa(a, AgentType)]
    min_inc_actual = isempty(all_incomes) ? 0.0 : minimum(all_incomes)
    max_inc_actual = isempty(all_incomes) ? 1.0 : (min_inc_actual == maximum(all_incomes) ? min_inc_actual + 1.0 : maximum(all_incomes))

    ## MODIFICA 1: Rimuovi i calcoli dinamici dei limiti per affitto e amenità.
    # all_amenities_values = filter(x -> !isnan(x), model.amenity_level)
    # min_amenity = ...
    # max_amenity = ...
    #
    # all_house_rents = Float64[]
    # ... (tutta la logica per calcolare min_rent_model e max_rent_model va cancellata) ...

    ## MODIFICA 2: Recupera i limiti FISSI dalle proprietà del modello.
    min_rent_plot, max_rent_plot = model.plot_rent_range
    min_amenity_plot, max_amenity_plot = model.plot_amenity_range
    
    # --- FUNZIONE COLORE AGENTI (invariata) ---
    function agent_color_by_income(agent)
        if !hasproperty(agent, :income_value) || !isa(agent.income_value, Real)
            return RGBA(0.5,0.5,0.5,1.0) # Grigio per default
        end
        range_inc = max_inc_actual - min_inc_actual
        normalized_income = range_inc > 1e-9 ? (agent.income_value - min_inc_actual) / range_inc : 0.5
        normalized_income = clamp(normalized_income, 0.0, 1.0)
        if agent isa LowIncome; return get_low_income_color_from_norm(normalized_income);
        elseif agent isa HighIncome; return get_high_income_color_from_norm(normalized_income);
        else; return RGBA(0.0, 0.0, 0.0, 1.0); end
    end

    # --- COSTRUZIONE DELLA COLORMAP DISCRETA (invariata) ---
    N_rent_bins = 20
    N_amenity_bins = 20    
    final_colormap = RGBA{Float64}[]
    # ... (tutta la costruzione della colormap rimane identica)
    id_empty = 1
    push!(final_colormap, RGBA(1.0, 1.0, 1.0, 1.0))
    base_r_low_rent = 0.7; base_g_low_rent = 0.7; base_b_low_rent = 1.0; 
    base_r_high_rent = 1.0; base_g_high_rent = 0.4; base_b_high_rent = 0.4; 
    house_rent_sub_colormap = RGBA{Float64}[]
    for i_bin in 0:(N_rent_bins-1)
        local norm_val_for_bin 
        if N_rent_bins == 1; norm_val_for_bin = 0.5; else; norm_val_for_bin = i_bin / (N_rent_bins - 1); end
        r_val = base_r_low_rent + norm_val_for_bin * (base_r_high_rent - base_r_low_rent)
        g_val = base_g_low_rent + norm_val_for_bin * (base_g_high_rent - base_g_low_rent)
        b_val = base_b_low_rent + norm_val_for_bin * (base_b_high_rent - base_b_low_rent)
        push!(final_colormap, RGBA(r_val, g_val, b_val, 1.0))
        push!(house_rent_sub_colormap, RGBA(r_val, g_val, b_val, 1.0))
    end
    offset_after_house_rent = id_empty + N_rent_bins
    base_r_dark_green = 0.1; base_g_dark_green = 0.3; base_b_dark_green = 0.1;
    base_r_light_green = 0.6; base_g_light_green = 1.0; base_b_light_green = 0.6;
    amenity_greens_sub_colormap = RGBA{Float64}[] 
    for i_bin in 0:(N_amenity_bins-1)
        local norm_val_for_bin 
        if N_amenity_bins == 1; norm_val_for_bin = 0.5; else; norm_val_for_bin = i_bin / (N_amenity_bins - 1); end
        r_val = base_r_dark_green + norm_val_for_bin * (base_r_light_green - base_r_dark_green)
        g_val = base_g_dark_green + norm_val_for_bin * (base_g_light_green - base_g_dark_green)
        b_val = base_b_dark_green + norm_val_for_bin * (base_b_light_green - base_b_dark_green)
        push!(final_colormap, RGBA(r_val, g_val, b_val, 1.0))
        push!(amenity_greens_sub_colormap, RGBA(r_val, g_val, b_val, 1.0))
    end
    offset_after_public = offset_after_house_rent + N_amenity_bins
    id_cbd = offset_after_public + 1
    push!(final_colormap, RGBA(0.2, 0.2, 0.2, 1.0))
    total_colormap_entries = length(final_colormap)


    # --- CREAZIONE MATRICE DEGLI ID PER LA COLORMAP ---
    patch_ids_for_colormap = zeros(Int, size(model.patch_type))
    try
        for i_cartesian_idx in CartesianIndices(model.patch_type)
            patch_sym = model.patch_type[i_cartesian_idx]
            if patch_sym == :house
                rent_val = model.capitalized_rent[i_cartesian_idx]
                
                ## MODIFICA 3: Normalizza l'affitto usando i limiti fissi.
                range_rent_plot = max_rent_plot - min_rent_plot
                normalized_rent = 0.0
                if !isnan(rent_val) && range_rent_plot > 1e-9
                    normalized_rent = (rent_val - min_rent_plot) / range_rent_plot
                end
                normalized_rent = clamp(normalized_rent, 0.0, 1.0)

                bin_index_rent = floor(Int, normalized_rent * (N_rent_bins - 1e-9))
                bin_index_rent = clamp(bin_index_rent, 0, N_rent_bins - 1)
                patch_ids_for_colormap[i_cartesian_idx] = id_empty + 1 + bin_index_rent
            
            elseif patch_sym == :public
                amenity_val = model.amenity_level[i_cartesian_idx]
                
                ## MODIFICA 4: Normalizza le amenità usando i limiti fissi.
                range_amenity_plot = max_amenity_plot - min_amenity_plot
                normalized_amenity = 0.0
                if !isnan(amenity_val) && range_amenity_plot > 1e-9
                    normalized_amenity = (amenity_val - min_amenity_plot) / range_amenity_plot
                end
                normalized_amenity = clamp(normalized_amenity, 0.0, 1.0)
                
                bin_index_amenity = floor(Int, normalized_amenity * (N_amenity_bins - 1e-9))
                bin_index_amenity = clamp(bin_index_amenity, 0, N_amenity_bins - 1)
                patch_ids_for_colormap[i_cartesian_idx] = offset_after_house_rent + 1 + bin_index_amenity
            
            elseif patch_sym == :empty
                patch_ids_for_colormap[i_cartesian_idx] = id_empty
            elseif patch_sym == :cbd
                patch_ids_for_colormap[i_cartesian_idx] = id_cbd
            else 
                patch_ids_for_colormap[i_cartesian_idx] = id_empty
            end
        end
    catch e; println("ERRORE durante creazione patch_ids_for_colormap:");showerror(stdout, e);println(); end
        
    if isempty(patch_ids_for_colormap); println("ERRORE CRITICO: patch_ids_for_colormap è vuota."); return; end

    # --- GENERAZIONE PLOT ABM (invariato) ---
    local fig_abm, ax_abm, abmobs_obj
    plot_successful = false
    # ... (la chiamata a abmplot rimane uguale) ...
    if patch_ids_for_colormap !== nothing && !isempty(patch_ids_for_colormap)
        try
            fig_abm, ax_abm, abmobs_obj = abmplot(
                model;
                agent_color = agent_color_by_income, agent_size = 10, agent_marker = :circle,
                heatarray = _ -> patch_ids_for_colormap,
                heatkwargs = ( colormap = final_colormap, colorrange = (1, total_colormap_entries) ),
                figure = (; resolution = (1200, 900)),
                add_colorbar = false
            )
            plot_successful = true
        catch e_plot; println("ERRORE durante abmplot: "); showerror(stdout, e_plot); println(); return; end
    else; println("AVVISO: Salto abmplot."); return; end

    # --- Blocco di codice per disegnare il rettangolo ---
    if plot_successful && isdefined(Main, :get_inner_city_bounds)
        try
            # Nota: la variabile degli assi si chiama 'ax_abm' in city_plot e city_plot_rent
            # e 'ax_pot_rent_detailed' in plot_potential_rent. Dobbiamo usare il nome corretto.
            
            # Recupera le coordinate del rettangolo
            r_start, r_end, c_start, c_end = get_inner_city_bounds(model)
            x_coords = [c_start - 0.5, c_end + 0.5, c_end + 0.5, c_start - 0.5, c_start - 0.5]
            y_coords = [r_start - 0.5, r_start - 0.5, r_end + 0.5, r_end + 0.5, r_start - 0.5]

            # Disegna le linee sull'asse corretto (ax_abm o ax_pot_rent_detailed)
            # La variabile 'ax' DEVE corrispondere a quella restituita da abmplot in quella specifica funzione.
            current_axis = @isdefined(ax_abm) ? ax_abm : ax_pot_rent_detailed
            Makie.lines!(current_axis, x_coords, y_coords, color = :black, linewidth = 2, linestyle = :dash)
        
        catch e_rect
            println("ERRORE disegno rettangolo Inner City: "); showerror(stdout, e_rect); println()
        end
    end

    # --- AGGIUNTA COLORBARS PERSONALIZZATE ---
    if plot_successful
        ## MODIFICA 5: Usa i limiti fissi nelle chiamate a Colorbar.
        Makie.Colorbar(fig_abm[1, 2], colormap = amenity_greens_sub_colormap, limits = (min_amenity_plot, max_amenity_plot), label = "Amenity Level", width = 30,labelsize = 30)
        Makie.Colorbar(fig_abm[1, 3], colormap = house_rent_sub_colormap, limits = (min_rent_plot, max_rent_plot), label = "Capitalized Rent", width = 30,labelsize = 30)
        
        # Le colorbar per il reddito degli agenti possono rimanere dinamiche (invariate)
        num_colorbar_samples = 50
        # ... (il resto della funzione rimane uguale) ...
        if isdefined(Main, :get_low_income_color_from_norm)
            low_income_cb_colors = [get_low_income_color_from_norm(n) for n in range(0, 1, length=num_colorbar_samples)]
            Makie.Colorbar(fig_abm[2, 1], colormap = low_income_cb_colors, limits = (min_inc_actual, max_inc_actual), label = "Low-income", vertical = false, height = 30, flipaxis = false,labelsize = 30 )
        end
        if isdefined(Main, :get_high_income_color_from_norm)
            high_income_cb_colors = [get_high_income_color_from_norm(n) for n in range(0, 1, length=num_colorbar_samples)]
            Makie.Colorbar(fig_abm[3, 1], colormap = high_income_cb_colors, limits = (min_inc_actual, max_inc_actual), label = "High-income", vertical = false, height = 30, flipaxis = false,labelsize = 30 )
        end
             
       #= Makie.colsize!(fig_abm.layout, 1, Makie.Relative(0.75))
        Makie.colsize!(fig_abm.layout, 2, Makie.Relative(0.125))
        Makie.colsize!(fig_abm.layout, 3, Makie.Relative(0.125))
        Makie.rowsize!(fig_abm.layout, 1, Makie.Relative(0.80))
        Makie.rowsize!(fig_abm.layout, 2, Makie.Relative(0.10))
        Makie.rowsize!(fig_abm.layout, 3, Makie.Relative(0.10))=#

        colsize!(fig_abm.layout, 1, Aspect(1, 1.0))

        Makie.colgap!(fig_abm.layout, 1, 30)
        Makie.colgap!(fig_abm.layout, 2, 30)
        Makie.rowgap!(fig_abm.layout, 1, 30)
        Makie.rowgap!(fig_abm.layout, 2, 30)

        display(fig_abm)
        println("Visualizzazione ABM con heatmap affitti e colorbar generata.")
    end
end



function plot_potential_rent(model::ABM)
    # --- CALCOLI PRELIMINARI PER LE SFUMATURE ---
    ## MODIFICA 1: Rimuovi i calcoli dinamici dei limiti.
    # # Per Potential Rent (case)
    # all_house_potential_rents = Float64[]
    # ... (logica per calcolare min_pot_rent_model e max_pot_rent_model cancellata) ...
    #
    # # Per Amenity Level (aree pubbliche)
    # all_amenities_values = filter(x -> !isnan(x), model.amenity_level)
    # ... (logica per calcolare min_amenity e max_amenity cancellata) ...

    ## MODIFICA 2: Recupera i limiti FISSI dalle proprietà del modello.
    min_pot_rent_plot, max_pot_rent_plot = model.plot_potential_rent_range
    min_amenity_plot, max_amenity_plot = model.plot_amenity_range

    # --- COSTRUZIONE DELLA COLORMAP DISCRETA (invariata) ---
    N_potential_rent_bins = 20
    N_amenity_bins = 20
    final_colormap = RGBA{Float64}[]
    # ... (tutta la costruzione della colormap rimane identica) ...
    id_empty = 1
    push!(final_colormap, RGBA(1.0, 1.0, 1.0, 1.0))
    base_r_low_pot = 0.7; base_g_low_pot = 0.9; base_b_low_pot = 1.0;
    base_r_high_pot = 1.0; base_g_high_pot = 0.8; base_b_high_pot = 0.4;
    potential_rent_sub_colormap = RGBA{Float64}[]
    for i_bin in 0:(N_potential_rent_bins-1)
        local norm_val_for_bin 
        if N_potential_rent_bins == 1; norm_val_for_bin = 0.5; else; norm_val_for_bin = i_bin / (N_potential_rent_bins - 1); end
        r_val = base_r_low_pot + norm_val_for_bin * (base_r_high_pot - base_r_low_pot)
        g_val = base_g_low_pot + norm_val_for_bin * (base_g_high_pot - base_g_low_pot)
        b_val = base_b_low_pot + norm_val_for_bin * (base_b_high_pot - base_b_low_pot)
        push!(final_colormap, RGBA(r_val, g_val, b_val, 1.0))
        push!(potential_rent_sub_colormap, RGBA(r_val, g_val, b_val, 1.0))
    end
    offset_after_house = id_empty + N_potential_rent_bins
    base_r_dark_green = 0.1; base_g_dark_green = 0.3; base_b_dark_green = 0.1;
    base_r_light_green = 0.6; base_g_light_green = 1.0; base_b_light_green = 0.6;
    amenity_greens_sub_colormap = RGBA{Float64}[] 
    for i_bin in 0:(N_amenity_bins-1)
        local norm_val_for_bin 
        if N_amenity_bins == 1; norm_val_for_bin = 0.5; else; norm_val_for_bin = i_bin / (N_amenity_bins - 1); end
        r_val = base_r_dark_green + norm_val_for_bin * (base_r_light_green - base_r_dark_green)
        g_val = base_g_dark_green + norm_val_for_bin * (base_g_light_green - base_g_dark_green)
        b_val = base_b_dark_green + norm_val_for_bin * (base_b_light_green - base_b_dark_green)
        current_green_color = RGBA(r_val, g_val, b_val, 1.0)
        push!(final_colormap, current_green_color)
        push!(amenity_greens_sub_colormap, current_green_color)
    end
    offset_after_public = offset_after_house + N_amenity_bins
    id_cbd = offset_after_public + 1
    push!(final_colormap, RGBA(0.2, 0.2, 0.2, 1.0))
    total_colormap_entries = length(final_colormap)


    # --- CREAZIONE MATRICE DEGLI ID PER LA COLORMAP ---
    patch_ids_for_colormap = zeros(Int, size(model.patch_type))
    try
        for i_cartesian_idx in CartesianIndices(model.patch_type)
            patch_sym = model.patch_type[i_cartesian_idx]
            if patch_sym == :house 
                pot_rent_val = model.potential_rent[i_cartesian_idx]
                
                ## MODIFICA 3: Normalizza l'affitto potenziale usando i limiti fissi.
                range_pot_rent_plot = max_pot_rent_plot - min_pot_rent_plot
                normalized_pot_rent = 0.0
                if !isnan(pot_rent_val) && range_pot_rent_plot > 1e-9
                    normalized_pot_rent = (pot_rent_val - min_pot_rent_plot) / range_pot_rent_plot
                end
                normalized_pot_rent = clamp(normalized_pot_rent, 0.0, 1.0)
                
                bin_index_pot_rent = floor(Int, normalized_pot_rent * (N_potential_rent_bins - 1e-9))
                bin_index_pot_rent = clamp(bin_index_pot_rent, 0, N_potential_rent_bins - 1)
                patch_ids_for_colormap[i_cartesian_idx] = id_empty + 1 + bin_index_pot_rent
            
            elseif patch_sym == :public
                amenity_val = model.amenity_level[i_cartesian_idx]

                ## MODIFICA 4: Normalizza le amenità usando i limiti fissi.
                range_amenity_plot = max_amenity_plot - min_amenity_plot
                normalized_amenity = 0.0
                if !isnan(amenity_val) && range_amenity_plot > 1e-9
                    normalized_amenity = (amenity_val - min_amenity_plot) / range_amenity_plot
                end
                normalized_amenity = clamp(normalized_amenity, 0.0, 1.0)

                bin_index_amenity = floor(Int, normalized_amenity * (N_amenity_bins - 1e-9))
                bin_index_amenity = clamp(bin_index_amenity, 0, N_amenity_bins - 1)
                patch_ids_for_colormap[i_cartesian_idx] = offset_after_house + 1 + bin_index_amenity
            
            elseif patch_sym == :empty
                patch_ids_for_colormap[i_cartesian_idx] = id_empty
            elseif patch_sym == :cbd
                patch_ids_for_colormap[i_cartesian_idx] = id_cbd
            else 
                patch_ids_for_colormap[i_cartesian_idx] = id_empty 
            end
        end
    catch e; println("ERRORE durante creazione patch_ids_for_colormap:"); showerror(stdout, e); println(); end
    
    if isempty(patch_ids_for_colormap); println("ERRORE CRITICO: patch_ids_for_colormap è vuota."); return; end

    # --- GENERAZIONE PLOT ABM (invariato) ---
    local fig_pot_rent_detailed, ax_pot_rent_detailed 
    plot_successful = false
    # ... (la chiamata ad abmplot rimane uguale) ...
    if patch_ids_for_colormap !== nothing && !isempty(patch_ids_for_colormap)
        try
            fig_pot_rent_detailed, ax_pot_rent_detailed, _ = abmplot(
                model;
                agent_size = 0, # Nasconde gli agenti
                heatarray = _ -> patch_ids_for_colormap,
                heatkwargs = ( colormap = final_colormap, colorrange = (1, total_colormap_entries) ),
                figure = (; resolution = (1000, 750)),
                axis = (; ),

                add_colorbar = false 
            )
            plot_successful = true 
        catch e_plot; println("ERRORE durante abmplot: "); showerror(stdout, e_plot); println(); return; end
    else; println("AVVISO: Salto abmplot."); return; end


    # --- Blocco di codice per disegnare il rettangolo ---
    if plot_successful && isdefined(Main, :get_inner_city_bounds)
        try
            # Nota: la variabile degli assi si chiama 'ax_abm' in city_plot e city_plot_rent
            # e 'ax_pot_rent_detailed' in plot_potential_rent. Dobbiamo usare il nome corretto.
            
            # Recupera le coordinate del rettangolo
            r_start, r_end, c_start, c_end = get_inner_city_bounds(model)
            x_coords = [c_start - 0.5, c_end + 0.5, c_end + 0.5, c_start - 0.5, c_start - 0.5]
            y_coords = [r_start - 0.5, r_start - 0.5, r_end + 0.5, r_end + 0.5, r_start - 0.5]

            # Disegna le linee sull'asse corretto (ax_abm o ax_pot_rent_detailed)
            # La variabile 'ax' DEVE corrispondere a quella restituita da abmplot in quella specifica funzione.
            current_axis = @isdefined(ax_abm) ? ax_abm : ax_pot_rent_detailed
            Makie.lines!(current_axis, x_coords, y_coords, color = :black, linewidth = 2, linestyle = :dash)
        
        catch e_rect
            println("ERRORE disegno rettangolo Inner City: "); showerror(stdout, e_rect); println()
        end
    end

    # --- AGGIUNTA COLORBARS PERSONALIZZATE ---
    if plot_successful
        ## MODIFICA 5: Usa i limiti fissi nelle chiamate a Colorbar.
        # Layout: Plot [1,1], CB Potential Rent [1,2], CB Amenity [1,3]

        cb_amenity_level = Colorbar(fig_pot_rent_detailed[1, 3], 
            colormap = amenity_greens_sub_colormap, 
            limits = (min_amenity_plot, max_amenity_plot), 
            label = "Amenity Level", width = 30, labelsize = 30)

        cb_potential_rent = Colorbar(fig_pot_rent_detailed[1, 2], 
            colormap = potential_rent_sub_colormap, 
            limits = (min_pot_rent_plot, max_pot_rent_plot),
            label = "Potential Rent", width = 30, labelsize=30)

        
        
        # Aggiustamenti Layout (invariato)
        colsize!(fig_pot_rent_detailed.layout, 1, Relative(0.80)) # Plot principale
        colsize!(fig_pot_rent_detailed.layout, 2, Relative(0.10)) # CB Affitto Potenziale
        colsize!(fig_pot_rent_detailed.layout, 3, Relative(0.10)) # CB Amenità
        
        colgap!(fig_pot_rent_detailed.layout, 1, 10) # Spazio tra plot e prima CB
        colgap!(fig_pot_rent_detailed.layout, 2, 10) # Spazio tra le due CB

        display(fig_pot_rent_detailed) 
        println("Visualizzazione Affitto Potenziale con sfondi dettagliati generata.")
    end
end



function get_vacant_house_positions(model::ABM)
    vacant_houses = NTuple{2, Int}[]
    dims = size(model.patch_type) # Ottieni le dimensioni una volta

    # Itera su tutte le posizioni nella griglia
    for r in 1:dims[1] # Righe
        for c in 1:dims[2] # Colonne
            current_pos = (r, c)
            # MODIFICA: Usa la funzione API ids_in_position
            # ids_in_position(current_pos, model.space) restituisce un iteratore degli ID degli agenti in quella posizione.
            # isempty() su questo iteratore ti dice se la posizione è vuota.
            if model.patch_type[r, c] == :house && isempty(ids_in_position(current_pos, model.space))
                push!(vacant_houses, current_pos)
            end
        end
    end
    return vacant_houses
end


function custom_sample_without_replacement(rng::Random.AbstractRNG, population::AbstractVector, k::Int)
    n = length(population)
    
    # Controlli sugli input
    if k < 0
        error("Il numero di elementi da campionare (k) non può essere negativo.")
    elseif k == 0
        return similar(population, 0) # Restituisce un vettore vuoto dello stesso tipo
    elseif k > n
        error("Il numero di elementi da campionare (k=$k) non può essere maggiore della dimensione della popolazione (n=$n).")
    end

    # Crea una copia degli indici della popolazione
    indices = collect(1:n)
    
    # Mescola gli indici. Questa è la parte che dipende ancora da Random.shuffle.
    # Se anche Random.shuffle dà problemi, il problema con il modulo Random è più serio.
    Random.shuffle!(rng, indices) # Mescola 'in-place' per efficienza se possibile, o usa Random.shuffle
    
    # Prendi i primi k indici mescolati
    selected_indices = view(indices, 1:k) # 'view' è efficiente, non crea una nuova copia subito
    
    # Restituisci gli elementi della popolazione originale corrispondenti a questi indici selezionati
    return population[selected_indices]
end

function get_inner_city_bounds(model::ABM)
    dims = model.grid_dims
    # Calcola il blocco centrale N/2 x N/2
    # Per una griglia 40x40, questo sarà (11, 30, 11, 30)
    r_start_inner = floor(Int, dims[1] / 4) + 1
    r_end_inner   = floor(Int, dims[1] * 3 / 4)
    c_start_inner = floor(Int, dims[2] / 4) + 1
    c_end_inner   = floor(Int, dims[2] * 3 / 4)
    return (r_start_inner, r_end_inner, c_start_inner, c_end_inner)
end

function is_in_inner_city(pos::NTuple{2, Int}, model::ABM)
    r_start, r_end, c_start, c_end = get_inner_city_bounds(model)
    return (r_start <= pos[1] <= r_end) && (c_start <= pos[2] <= c_end)
end

function run_simulation_phase_and_collect_data(
    input_model::ABM,
    n_steps_for_phase::Int,
    current_agent_step!::Function,
    current_model_step!::Function,
    initial_step_for_data::Int = 0,
    phase_label::String = "Fase Sconosciuta"
)

    println("--- Inizio $phase_label (Steps da $(initial_step_for_data + 1) a $(initial_step_for_data + n_steps_for_phase)) ---")
    
    # Dati agenti
    #=history_agent_step = Int[]
    history_agent_id = Int[]
    history_agent_type = Symbol[]
    history_agent_utility = Float64[]
    history_agent_pos = NTuple{2, Int}[]=#
    
    # Dati modello (esistenti)
    history_model_step = Int[]
    history_ratio_lh_inner = Float64[]
    history_ratio_lh_suburbs = Float64[]
    history_density_total_inner = Float64[]
    history_density_total_suburbs = Float64[]
    history_n_low_inner = Int[]
    history_n_high_inner = Int[]
    history_n_low_suburbs = Int[]
    history_n_high_suburbs = Int[]
    
    # Dati affitti
    history_mean_inner_city_rent = Float64[]
    history_median_inner_city_rent = Float64[]
    history_n_houses_inner_city_rent_calc = Int[]

    history_developer_status = Symbol[]

    # Metriche di reddito
    history_mean_income_inner_city = Float64[]
    history_median_income_inner_city = Float64[]

    # Metriche di segregazione
    history_neighbour_segregation_index = Float64[]
    history_amenity_segregation_index_low = Float64[]
    history_amenity_segregation_index_high = Float64[]

    history_agent_zero_sat = Int[]

    # --- NUOVI ARRAY STORICO PER DATI QUADRANTE ---
    history_quadrant_step = Int[]
    history_quadrant_coords_x = Int[]
    history_quadrant_coords_y = Int[]
    history_quadrant_mean_amenity = Float64[] # Amenity media pre-calcolata del quadrante
    history_quadrant_n_low = Int[]
    history_quadrant_n_high = Int[]

    #history_dissimilarity_index = Float64[]
    #history_exposure_index = Float64[]

    # --- FUNZIONE HELPER INTERNA PER LA RACCOLTA DEI DATI (incluse le metriche dei quadranti) ---
    function _collect_all_data_for_current_step(model_to_sample, current_step_number)
        
        #=
        # 1. Dati a livello di Agente
        current_step_zero_utility_count = 0
        for agent_obj in allagents(model_to_sample)
            push!(history_agent_step, current_step_number)
            push!(history_agent_id, agent_obj.id)
            push!(history_agent_type, agent_obj isa LowIncome ? :LowIncome : :HighIncome)
            agent_utility = calculate_total_utility(agent_obj, agent_obj.pos, model_to_sample)
            push!(history_agent_utility, agent_utility)
            push!(history_agent_pos, agent_obj.pos) # Salva la posizione
            
            if agent_utility == 0.0
                current_step_zero_utility_count += 1
            end
        end=#
        
        #push!(history_agent_zero_sat, current_step_zero_utility_count)

        # 2. Dati a livello di Modello
        metrics = calculate_urban_metrics(model_to_sample)
        push!(history_model_step, current_step_number)
        push!(history_ratio_lh_inner, metrics.ratio_lh_inner)
        push!(history_ratio_lh_suburbs, metrics.ratio_lh_suburbs)
        push!(history_density_total_inner, metrics.density_total_inner)
        push!(history_density_total_suburbs, metrics.density_total_suburbs)
        push!(history_n_low_inner, metrics.n_low_inner)
        push!(history_n_high_inner, metrics.n_high_inner)
        push!(history_n_low_suburbs, metrics.n_low_suburbs)
        push!(history_n_high_suburbs, metrics.n_high_suburbs)
        
        # MODIFICA: ACCESSO DIRETTO ALLA PROPRIETÀ developer_activity_status
        if hasproperty(model_to_sample, :developer_activity_status)
             push!(history_developer_status, model_to_sample.developer_activity_status)
        else
            push!(history_developer_status, :undefined) 
        end
        
        push!(history_mean_income_inner_city, metrics.mean_income_inner_city)
        push!(history_median_income_inner_city, metrics.median_income_inner_city)

        rent_metrics = calculate_inner_city_rent_metric(model_to_sample)
        push!(history_mean_inner_city_rent, rent_metrics.mean_inner_city_rent)
        push!(history_median_inner_city_rent, rent_metrics.median_inner_city_rent)
        push!(history_n_houses_inner_city_rent_calc, rent_metrics.n_houses_inner_city_for_rent_calc)
        
        push!(history_neighbour_segregation_index, calculate_neighbour_segregation_index(model_to_sample))
        low_amenity_score, high_amenity_score = calculate_amenity_segregation_index(model_to_sample)
        push!(history_amenity_segregation_index_low, low_amenity_score)
        push!(history_amenity_segregation_index_high, high_amenity_score)

        # 3. --- Raccolta Dati per Quadrante ---
        # MODIFICA: ACCESSO DIRETTO ALLA PROPRIETÀ mean_quadrant_amenities_map
        mean_quadrant_amenities_map = model_to_sample.mean_quadrant_amenities_map # <--- MODIFICATO
        
        # MODIFICA: ACCESSO DIRETTO ALLA PROPRIETÀ grid_dims (se la tua initialize_model la salva così)
        # Se invece model.grid_dims NON esiste e la dimensione della griglia è sempre la stessa,
        # puoi usare la variabile globale 'abm_grid_dims' o 'dims' passata al modello.
        # Oppure, se 'grid_dims' è una proprietà del modello, la recuperi direttamente:
        grid_dims = model_to_sample.grid_dims # <--- MODIFICATO, assuming it's a direct property
                                              # (it's what you put into `properties[:grid_dims]`)
        
        current_step_quadrant_counts = Dict{NTuple{2, Int}, Dict{Symbol, Int}}()
        
        quadrant_size_x = div(grid_dims[1], 4)
        quadrant_size_y = div(grid_dims[2], 4)

        for agent_obj in allagents(model_to_sample)
            qx = ceil(Int, agent_obj.pos[1] / quadrant_size_x)
            qy = ceil(Int, agent_obj.pos[2] / quadrant_size_y)
            quadrant_coords = (qx, qy)

            if !haskey(current_step_quadrant_counts, quadrant_coords)
                current_step_quadrant_counts[quadrant_coords] = Dict(:n_low => 0, :n_high => 0)
            end

            if agent_obj isa LowIncome
                current_step_quadrant_counts[quadrant_coords][:n_low] += 1
            elseif agent_obj isa HighIncome
                current_step_quadrant_counts[quadrant_coords][:n_high] += 1
            end
        end

        all_quadrant_keys = sort(collect(keys(mean_quadrant_amenities_map)))
        
        for q_coords in all_quadrant_keys
            n_low = get(current_step_quadrant_counts, q_coords, Dict(:n_low => 0))[:n_low]
            n_high = get(current_step_quadrant_counts, q_coords, Dict(:n_high => 0))[:n_high]
            mean_q_amenity = get(mean_quadrant_amenities_map, q_coords, 0.0)

            push!(history_quadrant_step, current_step_number)
            push!(history_quadrant_coords_x, q_coords[1])
            push!(history_quadrant_coords_y, q_coords[2])
            push!(history_quadrant_mean_amenity, mean_q_amenity)
            push!(history_quadrant_n_low, n_low)
            push!(history_quadrant_n_high, n_high)
        end


    end # Fine di _collect_all_data_for_current_step

    # --- Raccogli dati per lo stato iniziale di questa fase ---
    if n_steps_for_phase >= 0
        _collect_all_data_for_current_step(input_model, initial_step_for_data)
    end

    # --- Ciclo di simulazione per questa fase ---
    for s_phase_loop_idx in 1:n_steps_for_phase
        absolute_step = initial_step_for_data + s_phase_loop_idx
        
        step!(input_model, current_agent_step!, current_model_step!)
        
        _collect_all_data_for_current_step(input_model, absolute_step)
        
        if s_phase_loop_idx % 10 == 0 || s_phase_loop_idx == n_steps_for_phase
            println("Completato step di $phase_label: $s_phase_loop_idx / $n_steps_for_phase (Step assoluto: $absolute_step)")
        end
    end
    println("--- Fine $phase_label ---")

    # --- Creazione dei DataFrame finali ---
    agent_data_df = DataFrame(
        #step=history_agent_step,
        #id=history_agent_id,
        #agent_type=history_agent_type,
        #utility=history_agent_utility,
        #pos=history_agent_pos
    )
    
    model_metrics_df = DataFrame(
        step=history_model_step,
        ratio_lh_inner=history_ratio_lh_inner,
        ratio_lh_suburbs=history_ratio_lh_suburbs,
        density_total_inner=history_density_total_inner,
        density_total_suburbs=history_density_total_suburbs,
        n_low_inner=history_n_low_inner,
        n_high_inner=history_n_high_inner,
        n_low_suburbs=history_n_low_suburbs,
        n_high_suburbs=history_n_high_suburbs,
        developer_status=history_developer_status,
        mean_inner_city_rent=history_mean_inner_city_rent,
        median_inner_city_rent=history_median_inner_city_rent,
        n_houses_inner_city_rent_calc=history_n_houses_inner_city_rent_calc,
        mean_income_inner_city=history_mean_income_inner_city,
        median_income_inner_city=history_median_income_inner_city,
        neighbour_segregation_index=history_neighbour_segregation_index,
        amenity_segregation_index_low=history_amenity_segregation_index_low,
        amenity_segregation_index_high=history_amenity_segregation_index_high,
        #n_zero_utility = history_agent_zero_sat,

        # --- MODIFICA 3: Aggiungi le nuove colonne al DataFrame finale ---
        #dissimilarity_index = history_dissimilarity_index,
        #exposure_index = history_exposure_index
    )

    # --- NUOVO: DataFrame per i dati dei quadranti ---
    quadrant_data_df = DataFrame(
        step=history_quadrant_step,
        quadrant_coords_x=history_quadrant_coords_x,
        quadrant_coords_y=history_quadrant_coords_y,
        mean_quadrant_amenity=history_quadrant_mean_amenity,
        n_low_quadrant=history_quadrant_n_low,
        n_high_quadrant=history_quadrant_n_high
    )
    
    # --- LA FUNZIONE ORA RESTITUISCE 3 DATAFRAME E IL MODELLO ---
    return agent_data_df, model_metrics_df, quadrant_data_df, input_model
end


function calculate_urban_metrics(model::ABM)
    # --- Inizializzazione contatori e accumulatori ---
    n_low_inner, n_high_inner, n_total_agents_inner = 0, 0, 0
    n_low_suburbs, n_high_suburbs, n_total_agents_suburbs = 0, 0, 0
    n_house_patches_inner, n_house_patches_suburbs = 0, 0
    
    # NUOVI: Liste per i redditi dell'inner city
    incomes_inner_city = Float64[]

    inner_city_region_bounds = get_inner_city_bounds(model)

    # --- Conteggio patch 'house' (invariato) ---
    for r_idx in 1:model.grid_dims[1], c_idx in 1:model.grid_dims[2]
        current_patch_pos = (r_idx, c_idx)
        if model.patch_type[current_patch_pos...] == :house
            if is_in_inner_city(current_patch_pos, model)
                n_house_patches_inner += 1
            else
                n_house_patches_suburbs += 1
            end
        end
    end
    
    # --- Conteggio Agenti e Raccolta Redditi ---
    for agent_instance in allagents(model)
        # Considera solo agenti su patch di tipo ':house'
        if model.patch_type[agent_instance.pos...] == :house
            if is_in_inner_city(agent_instance.pos, model)
                n_total_agents_inner += 1
                push!(incomes_inner_city, agent_instance.income_value) # NUOVO: Raccogli reddito
                if agent_instance isa LowIncome
                    n_low_inner += 1
                else # HighIncome
                    n_high_inner += 1
                end
            else 
                n_total_agents_suburbs += 1
                if agent_instance isa LowIncome
                    n_low_suburbs += 1
                else # HighIncome
                    n_high_suburbs += 1
                end
            end
        end
    end

    # --- Calcolo Metriche Finali ---
    ratio_lh_inner = n_high_inner > 0 ? Float64(n_low_inner) / n_high_inner : NaN
    ratio_lh_suburbs = n_high_suburbs > 0 ? Float64(n_low_suburbs) / n_high_suburbs : NaN
    
    density_total_inner = n_house_patches_inner > 0 ? Float64(n_total_agents_inner) / n_house_patches_inner : 0.0
    density_total_suburbs = n_house_patches_suburbs > 0 ? Float64(n_total_agents_suburbs) / n_house_patches_suburbs : 0.0

    # NUOVO: Calcolo statistiche reddito per l'inner city
    mean_income_inner_city = !isempty(incomes_inner_city) ? mean(incomes_inner_city) : NaN
    median_income_inner_city = !isempty(incomes_inner_city) ? median(incomes_inner_city) : NaN

    return (
        # Conteggi e metriche esistenti
        n_low_inner = n_low_inner, n_high_inner = n_high_inner, 
        n_low_suburbs = n_low_suburbs, n_high_suburbs = n_high_suburbs,
        ratio_lh_inner = ratio_lh_inner,
        ratio_lh_suburbs = ratio_lh_suburbs,
        density_total_inner = density_total_inner,
        density_total_suburbs = density_total_suburbs,
        # NUOVE METRICHE
        mean_income_inner_city = mean_income_inner_city,
        median_income_inner_city = median_income_inner_city
    )
end


function calculate_inner_city_rent_metric(model::ABM)
    inner_city_rents = Float64[]

    # Itera su tutte le posizioni della griglia
    for r_idx in 1:model.grid_dims[1], c_idx in 1:model.grid_dims[2]
        current_pos = (r_idx, c_idx)
        
        # Controlla se la patch è una casa nell'inner city
        if model.patch_type[current_pos...] == :house && is_in_inner_city(current_pos, model)
            # --- MODIFICA CHIAVE: Controlla che la casa sia privata ---
            if model.management_type[current_pos...] == :private
                rent_val = model.capitalized_rent[current_pos...]
                if !isnan(rent_val)
                    push!(inner_city_rents, rent_val)
                end
            end
        end
    end

    # Calcola sia la media che la mediana
    mean_rent = !isempty(inner_city_rents) ? mean(inner_city_rents) : NaN
    median_rent = !isempty(inner_city_rents) ? median(inner_city_rents) : NaN # <-- NUOVO

    return (
        mean_inner_city_rent = mean_rent,
        median_inner_city_rent = median_rent, # <-- NUOVO
        n_houses_inner_city_for_rent_calc = length(inner_city_rents)
    )
end


function calculate_neighbour_segregation_index(model::ABM)
    if nagents(model) == 0
        return 1.0 
    end

    total_neighbor_links = 0
    same_type_links = 0

    for agent in allagents(model)
        for neighbor in nearby_agents(agent, model) 
            total_neighbor_links += 1
            if typeof(agent) == typeof(neighbor)
                same_type_links += 1
            end
        end
    end

    if total_neighbor_links == 0
        return 1.0
    end

    return same_type_links / total_neighbor_links
end

function calculate_amenity_segregation_index(model::ABM)
    if nagents(model) == 0
        return 1.0 
    end

    raw_score = model.raw_amenity_scores_matrix
    overall_min_score = model.global_min_raw_amenity_score
    overall_max_score = model.global_max_raw_amenity_score
    range_amenity = overall_max_score - overall_min_score

    low_income_amenity_exposure = []
    high_income_amenity_exposure = []

    for agent in allagents(model)
        if agent isa LowIncome
            push!(low_income_amenity_exposure, raw_score[agent.pos...])
        elseif agent isa HighIncome
            push!(high_income_amenity_exposure, raw_score[agent.pos...])
        end
    end

    high_amenity_segregation_index = (mean(high_income_amenity_exposure)-overall_min_score)/range_amenity
    low_amenity_segregation_index = (mean(low_income_amenity_exposure)-overall_min_score)/range_amenity

    return low_amenity_segregation_index, high_amenity_segregation_index
end



function plot_single_city_comparison_simple(
    model_before::ABM, 
    model_after::ABM, 
    plot_type::Symbol;
    title_fontsize::Int = 28
)
    # --- 1. Preparazione dei dati (invariata) ---
    title_fragment = ""
    primary_value_field_symbol = :house_age 
    plot_range_field_primary_symbol = :plot_age_range

    if plot_type == :age
        title_fragment = "House Age"
        primary_value_field_symbol = :house_age
        plot_range_field_primary_symbol = :plot_age_range
    elseif plot_type == :cap_rent
        title_fragment = "Capitalized Rent"
        primary_value_field_symbol = :capitalized_rent
        plot_range_field_primary_symbol = :plot_rent_range
    elseif plot_type == :pot_rent
        title_fragment = "Potential Rent"
        primary_value_field_symbol = :potential_rent
        plot_range_field_primary_symbol = :plot_potential_rent_range
    end

    println("Generazione plot di confronto per: $title_fragment...")
    
    # Definisci le funzioni helper per i colori e le heatmap (come prima)
    ac_before = _get_agent_color_func(model_before)
    ac_after = _get_agent_color_func(model_after)
    vis_data_before = _prepare_heatmap_visuals(model_before, plot_type, primary_value_field_symbol, plot_range_field_primary_symbol)
    vis_data_after = _prepare_heatmap_visuals(model_after, plot_type, primary_value_field_symbol, plot_range_field_primary_symbol)

    # --- 2. Creazione della Figura con Layout Verticale Semplice ---
    fig = Figure(size = (1500, 800))

    # --- Plot "Prima" ---
    ax_before = Axis(fig[1, 1], 
                     title = "$title_fragment (Before, Step $(model_before.step))", 
                     titlesize = title_fontsize, 
                     aspect = DataAspect())
    
    abmplot!(ax_before, model_before; 
             agent_color=ac_before, agent_size=12,
             heatarray=vis_data_before.heatarray, heatkwargs=vis_data_before.heatkwargs,
             add_colorbar=false)
    hidedecorations!(ax_before)
    r_start, r_end, c_start, c_end = get_inner_city_bounds(model_before)
    lines!(ax_before, [c_start - 0.5, c_end + 0.5, c_end + 0.5, c_start - 0.5, c_start - 0.5], 
           [r_start - 0.5, r_start - 0.5, r_end + 0.5, r_end + 0.5, r_start - 0.5], 
           color = :black, linewidth = 2, linestyle = :dash)

    # --- Plot "Dopo" ---
    ax_after = Axis(fig[1, 2], 
                    title = "$title_fragment (After, Step $(model_after.step))", 
                    titlesize = title_fontsize, 
                    aspect = DataAspect())
                    
    abmplot!(ax_after, model_after;
             agent_color=ac_after, agent_size=12,
             heatarray=vis_data_after.heatarray, heatkwargs=vis_data_after.heatkwargs,
             add_colorbar=false)
    hidedecorations!(ax_after)
    lines!(ax_after, [c_start - 0.5, c_end + 0.5, c_end + 0.5, c_start - 0.5, c_start - 0.5],
           [r_start - 0.5, r_start - 0.5, r_end + 0.5, r_end + 0.5, r_start - 0.5],
           color = :black, linewidth = 2, linestyle = :dash)
    
    # --- 3. Aggiungi un piccolo spazio tra i due plot ---
    colgap!(fig.layout, 1, 50)

    #display(fig)
    return fig
end


function _get_agent_color_func(model_state::ABM)
    all_incomes_list = [a.income_value for a in allagents(model_state) if hasproperty(a, :income_value) && isa(a.income_value, Real)]
    min_inc = isempty(all_incomes_list) ? 0.0 : minimum(all_incomes_list)
    max_inc = isempty(all_incomes_list) ? 1.0 : (min_inc == maximum(all_incomes_list) ? min_inc + 1.0 : maximum(all_incomes_list))
    return function agent_color_by_income(agent)
        if !hasproperty(agent, :income_value) || !isa(agent.income_value, Real); return RGBA(0.5,0.5,0.5,1.0); end
        range_inc = max_inc - min_inc
        norm_inc = range_inc > 1e-9 ? (agent.income_value - min_inc) / range_inc : 0.5
        norm_inc = clamp(norm_inc, 0.0, 1.0)
        if agent isa LowIncome; return get_low_income_color_from_norm(norm_inc);
        elseif agent isa HighIncome; return get_high_income_color_from_norm(norm_inc);
        else; return RGBA(0.0,0.0,0.0,1.0); end
    end
end

function _prepare_heatmap_visuals(model_for_ranges_and_config::ABM, p_type::Symbol, 
                                    p_value_field_symbol::Symbol, p_range_field_symbol::Symbol)
    N_primary_bins = 20
    N_amenity_bins = 20
    
    min_primary_plot, max_primary_plot = getproperty(model_for_ranges_and_config, p_range_field_symbol)
    min_amenity_plot, max_amenity_plot = model_for_ranges_and_config.plot_amenity_range

    primary_sub_colormap = Makie.ColorSchemes.viridis.colors
    if p_type == :age
        base_r_yellow=1.0; base_g_yellow=0.95; base_b_yellow=0.7; base_r_orange=0.9; base_g_orange=0.45; base_b_orange=0.0;
        primary_sub_colormap = [RGBA(base_r_yellow + (i/(N_primary_bins-1))*(base_r_orange-base_r_yellow), base_g_yellow + (i/(N_primary_bins-1))*(base_g_orange-base_g_yellow), base_b_yellow + (i/(N_primary_bins-1))*(base_b_orange-base_b_yellow), 1.0) for i in 0:(N_primary_bins-1)]
    elseif p_type == :cap_rent
        base_r_low_rent=0.7; base_g_low_rent=0.7; base_b_low_rent=1.0; base_r_high_rent=1.0; base_g_high_rent=0.4; base_b_high_rent=0.4;
        primary_sub_colormap = [RGBA(base_r_low_rent + (i/(N_primary_bins-1))*(base_r_high_rent-base_r_low_rent), base_g_low_rent + (i/(N_primary_bins-1))*(base_g_high_rent-base_g_low_rent), base_b_low_rent + (i/(N_primary_bins-1))*(base_b_high_rent-base_b_low_rent), 1.0) for i in 0:(N_primary_bins-1)]
    elseif p_type == :pot_rent
        base_r_low_pot=0.7; base_g_low_pot=0.9; base_b_low_pot=1.0; base_r_high_pot=1.0; base_g_high_pot=0.8; base_b_high_pot=0.4;
        primary_sub_colormap = [RGBA(base_r_low_pot + (i/(N_primary_bins-1))*(base_r_high_pot-base_r_low_pot), base_g_low_pot + (i/(N_primary_bins-1))*(base_g_high_pot-base_g_low_pot), base_b_low_pot + (i/(N_primary_bins-1))*(base_b_high_pot-base_b_low_pot), 1.0) for i in 0:(N_primary_bins-1)]
    end

    base_r_dark_green=0.1; base_g_dark_green=0.3; base_b_dark_green=0.1; base_r_light_green=0.6; base_g_light_green=1.0; base_b_light_green=0.6;
    amenity_sub_colormap = [RGBA(base_r_dark_green + (i/(N_amenity_bins-1))*(base_r_light_green-base_r_dark_green), base_g_dark_green + (i/(N_amenity_bins-1))*(base_g_light_green-base_g_dark_green), base_b_dark_green + (i/(N_amenity_bins-1))*(base_b_light_green-base_b_dark_green), 1.0) for i in 0:(N_amenity_bins-1)]

    # --- Aggiunta dei colori per i tipi di patch speciali e public housing ---
    public_housing_color = RGBA(1.0, 1.0, 1.0, 1.0) # Bianco per Public Housing
    empty_color = RGBA(0.8, 0.8, 0.8, 1.0) # Grigio chiaro per Empty
    cbd_color = RGBA(0.2, 0.2, 0.2, 1.0) # Grigio scuro per CBD

    # Composizione della colormap finale
    # L'ordine degli ID deve corrispondere all'ordine in cui li pushamo
    # ID 1: Empty
    # ID 2...N_primary_bins+1: House values (primary_sub_colormap)
    # ID N_primary_bins+2...N_primary_bins+N_amenity_bins+1: Public values (amenity_sub_colormap)
    # ID N_primary_bins+N_amenity_bins+2: CBD
    # ID N_primary_bins+N_amenity_bins+3: Public Housing
    final_combined_colormap = [
        empty_color;
        primary_sub_colormap;
        amenity_sub_colormap;
        cbd_color;
        public_housing_color
    ]

    # Definisci gli ID numerici per ciascun tipo di patch per `patch_ids`
    id_empty = 1
    id_first_primary = id_empty + 1 # Inizia dopo empty
    id_first_amenity = id_first_primary + N_primary_bins # Inizia dopo i primary bins
    id_cbd = id_first_amenity + N_amenity_bins # Inizia dopo gli amenity bins
    id_public_housing = id_cbd + 1 # Inizia dopo CBD

    total_entries = length(final_combined_colormap)

    heatarray_func = function(model_from_abmplot::ABM)
        patch_type_matrix = model_from_abmplot.patch_type
        management_type_matrix = model_from_abmplot.management_type # Accesso alla matrice management_type
        patch_ids = zeros(Int, size(patch_type_matrix))
        primary_value_matrix = getproperty(model_from_abmplot, p_value_field_symbol)
        amenity_level_matrix = model_from_abmplot.amenity_level
        
        for idx in CartesianIndices(patch_type_matrix)
            patch_sym = patch_type_matrix[idx]
            mgmt_sym = management_type_matrix[idx] # Recupera il management_type
            
            if patch_sym == :house # Se è una casa, dobbiamo distinguere per gestione
                if mgmt_sym == :public # Se è una casa di public housing
                    patch_ids[idx] = id_public_housing # Assegna il colore public housing (bianco)
                else # È una casa privata (:private), usa il suo colore basato sul valore primario
                    primary_val = primary_value_matrix[idx]
                    range_primary = max_primary_plot - min_primary_plot
                    norm_primary = (!isnan(primary_val) && range_primary > 1e-9) ? clamp((primary_val - min_primary_plot) / range_primary, 0.0, 1.0) : 0.0
                    bin_idx_primary = clamp(floor(Int, norm_primary * (N_primary_bins - 1e-9)), 0, N_primary_bins - 1)
                    patch_ids[idx] = id_first_primary + bin_idx_primary # ID corretto per i primary values
                end
            elseif patch_sym == :public # Se è un'area pubblica
                amenity_val = amenity_level_matrix[idx]
                range_amenity = max_amenity_plot - min_amenity_plot
                norm_amenity = (!isnan(amenity_val) && range_amenity > 1e-9) ? clamp((amenity_val - min_amenity_plot) / range_amenity, 0.0, 1.0) : 0.0
                bin_idx_amenity = clamp(floor(Int, norm_amenity * (N_amenity_bins - 1e-9)), 0, N_amenity_bins - 1)
                patch_ids[idx] = id_first_amenity + bin_idx_amenity # ID corretto per i amenity values
            elseif patch_sym == :empty; patch_ids[idx] = id_empty;
            elseif patch_sym == :cbd; patch_ids[idx] = id_cbd;
            else; patch_ids[idx] = id_empty; end # Fallback
        end
        return patch_ids
    end
    
    return (heatarray=heatarray_func, 
            heatkwargs=(colormap=final_combined_colormap, colorrange=(1, total_entries)),
            primary_cmap_details=(cmap=primary_sub_colormap, lims=(min_primary_plot, max_primary_plot)),
            amenity_cmap_details=(cmap=amenity_sub_colormap, lims=(min_amenity_plot, max_amenity_plot)),
            public_housing_color = public_housing_color # Passa il colore per la legenda
            )
end


function plot_income_trends_inner_city(data_df::DataFrame; title_suffix::String="")
    if isempty(data_df); println("DataFrame for Income Trends is empty."); return nothing; end

    is_aggregated = :mean_median_income_inner_city in Symbol.(names(data_df))
    col_median = is_aggregated ? :mean_median_income_inner_city : :median_income_inner_city
    sem_median = is_aggregated && (:sem_median_income_inner_city in Symbol.(names(data_df))) ? data_df.sem_median_income_inner_city : nothing

    if !(col_median in Symbol.(names(data_df))); println("Required data column is missing."); return nothing; end
    
    valid_data = filter(row -> !ismissing(row[col_median]) && !isnan(row[col_median]), data_df)
    if isempty(valid_data); println("No valid data to plot."); return nothing; end

    # --- Plot Creation ---
    plt_income = Plots.plot(
        legend = :topleft, # <-- Legend position
        title = "Inner City Median Income Evolution" * title_suffix,
        xlabel = "Simulation Step",
        ylabel = "Median Inner City Income (k\$)"
    )

    Plots.plot!(plt_income, valid_data.step, valid_data[!, col_median],
        label = "Median Income (Inner City)",
        ribbon = is_aggregated ? valid_data.sem_median_income_inner_city : nothing,
        linewidth = 2.5,
        marker = :circle
    )

    # --- Adaptive Y-axis logic ---
    min_val, max_val = extrema(valid_data[!, col_median])
    y_margin = (max_val - min_val) * 0.1
    Plots.ylims!(plt_income, (max(0, min_val - y_margin), max_val + y_margin))
    
    return plt_income
end



function perform_public_housing_conversion!(model::ABM)
    println("  [Public Housing] Inizio processo di conversione delle case...")

    # 1. Identifica le case private esistenti che possono essere candidate alla conversione.
    candidate_house_positions = NTuple{2, Int}[]
    
    total_initial_private_houses = model.total_initial_houses_count
    num_to_convert_target = round(Int, total_initial_private_houses * model.public_housing_fraction)

    for x in 1:model.grid_dims[1], y in 1:model.grid_dims[2]
        pos = (x, y)
        
        if model.patch_type[x, y] == :house && model.management_type[x, y] == :private && !(pos in model.renovated_houses_history)
            
            occupants_at_pos = ids_in_position(pos, model)
            
            if isempty(occupants_at_pos)
                push!(candidate_house_positions, pos)
            else
                agent_id = first(occupants_at_pos)
                agent = model[agent_id]
                
                if agent.income_value < model.public_housing_income_threshold
                    push!(candidate_house_positions, pos)
                end
            end
        end
    end

    if isempty(candidate_house_positions)
        println("  [Public Housing] Nessuna casa candidata (privata, non gentrificata, vuota/low-income) disponibile per la conversione. La politica non convertirà case.")
        return
    end

    num_to_convert = min(num_to_convert_target, length(candidate_house_positions))
    
    if num_to_convert == 0 && num_to_convert_target > 0
        println("  [Public Housing] Nonostante un target di $num_to_convert_target case, nessuna casa idonea trovata per la conversione.")
        return
    elseif num_to_convert == 0
        println("  [Public Housing] Nessuna casa convertita (target o disponibilità insufficiente di candidati).")
        return
    end

    converted_positions = NTuple{2, Int}[]
    if num_to_convert > 0
        converted_positions = StatsBase.sample(model.rng, candidate_house_positions, num_to_convert, replace=false)
    end
    
    println("  [Public Housing] Conversione di $(length(converted_positions)) case a Public Housing (su un target di $num_to_convert_target).")

    # Nuova soglia di età per la ristrutturazione da parte del governo
    AGE_THRESHOLD_FOR_GOV_RENOVATION = 40.0 # Puoi rendere questo un parametro del modello se vuoi sperimentare
    AGE_AFTER_GOV_RENOVATION = 40.0 # L'età a cui il governo riporta le case molto vecchie

    # 4. Applica la conversione alle case selezionate con logica di invecchiamento/mantenimento
    for pos in converted_positions
        model.management_type[pos...] = :public # Cambia il tipo di gestione
        model.capitalized_rent[pos...] = model.public_housing_rent # Imposta l'affitto fisso

        current_age_at_conversion = model.house_age[pos...]
        
        # Logica per l'età e il noise:
        # Le case di public housing non invecchieranno più automaticamente (gestito nel model_step_dinamica!)
        # e il loro noise sarà cristallizzato o azzerato.
        
        if current_age_at_conversion >= AGE_THRESHOLD_FOR_GOV_RENOVATION
            # Se la casa è molto vecchia, il governo la ristruttura a un'età più ragionevole
            model.house_age[pos...] = AGE_AFTER_GOV_RENOVATION
            println("    - Casa a $pos (età: $(round(current_age_at_conversion, digits=1))) ristrutturata a $(AGE_AFTER_GOV_RENOVATION).")
        else
            # Altrimenti, l'età viene "cristallizzata" a quella attuale al momento dell'acquisizione.
            # Non facciamo nulla qui perché l'età non verrà più modificata automaticamente dopo.
            println("    - Casa a $pos (età: $(round(current_age_at_conversion, digits=1))) età cristallizzata.")
        end
        
        # Il noise sulla condizione viene azzerato o cristallizzato se non vuoi che influenzi più la qualità
        # Azzeriamo per semplicità, simulando una manutenzione costante.
        model.house_condition_noise[pos...] = 0.0 
    end
    println("  [Public Housing] Processo di conversione completato.")
end


function perform_public_housing_conversion_by_rent_gap!(model::ABM)
    println("  [Public Housing - Strategia Rent Gap] Inizio processo di conversione delle case...")

    # 1. Identifica le case private esistenti che possono essere candidate alla conversione.
    # Ora raccoglieremo anche il rent_gap per ogni candidata.
    candidate_houses_with_gaps = [] # Useremo un array di NamedTuple per conservare gap e posizione
    
    total_initial_private_houses = model.total_initial_houses_count
    num_to_convert_target = round(Int, total_initial_private_houses * model.public_housing_fraction)

    for x in 1:model.grid_dims[1], y in 1:model.grid_dims[2]
        pos = (x, y)
        
        if model.patch_type[x, y] == :house && model.management_type[x, y] == :private && !(pos in model.renovated_houses_history)
            
            occupants_at_pos = ids_in_position(pos, model)
            is_eligible_by_occupant = false
            if !isempty(occupants_at_pos)
                agent_id = first(occupants_at_pos)
                agent = model[agent_id]
                if agent.income_value < model.public_housing_income_threshold
                    is_eligible_by_occupant = true
                end
            end

            # Se la casa è vuota O è occupata da un Low Income idoneo
            if isempty(occupants_at_pos) || is_eligible_by_occupant
                pr = model.potential_rent[pos...]
                cr = model.capitalized_rent[pos...]
                
                # Calcola il rent gap. Se non è una casa valida, o PR <= CR, il gap è 0 o negativo
                rent_gap_val = !isnan(pr) && !isnan(cr) ? (pr - cr) : 0.0
                
                push!(candidate_houses_with_gaps, (gap = rent_gap_val, position = pos))
            end
        end
    end

    if isempty(candidate_houses_with_gaps)
        println("  [Public Housing - Strategia Rent Gap] Nessuna casa candidata (privata, non gentrificata, vuota/low-income) disponibile per la conversione. La politica non convertirà case.")
        return
    end

    # 2. Ordina le candidate per rent gap (dal più alto al più basso)
    sort!(candidate_houses_with_gaps, by = x -> x.gap, rev = true)

    # 3. Determina quante case convertire effettivamente
    num_to_convert = min(num_to_convert_target, length(candidate_houses_with_gaps))
    
    if num_to_convert == 0 && num_to_convert_target > 0
        println("  [Public Housing - Strategia Rent Gap] Nonostante un target di $num_to_convert_target case, nessuna casa idonea trovata per la conversione.")
        return
    elseif num_to_convert == 0
        println("  [Public Housing - Strategia Rent Gap] Nessuna casa convertita (target o disponibilità insufficiente di candidati).")
        return
    end

    # 4. Prendi le prime `num_to_convert` case con il rent gap più alto
    converted_positions_by_rent_gap = [x.position for x in candidate_houses_with_gaps[1:num_to_convert]]
    
    println("  [Public Housing - Strategia Rent Gap] Conversione di $(length(converted_positions_by_rent_gap)) case a Public Housing (su un target di $num_to_convert_target).")

    # Nuova soglia di età per la ristrutturazione da parte del governo
    AGE_THRESHOLD_FOR_GOV_RENOVATION = 40.0 # Puoi rendere questo un parametro del modello se vuoi sperimentare
    AGE_AFTER_GOV_RENOVATION = 40.0 # L'età a cui il governo riporta le case molto vecchie

    # 5. Applica la conversione alle case selezionate
    for pos in converted_positions_by_rent_gap
        model.management_type[pos...] = :public # Cambia il tipo di gestione
        model.capitalized_rent[pos...] = model.public_housing_rent # Imposta l'affitto fisso

        current_age_at_conversion = model.house_age[pos...]
        
        if current_age_at_conversion >= AGE_THRESHOLD_FOR_GOV_RENOVATION
            model.house_age[pos...] = AGE_AFTER_GOV_RENOVATION
            println("    - Casa a $pos (età: $(round(current_age_at_conversion, digits=1))) ristrutturata a $(AGE_AFTER_GOV_RENOVATION).")
        else
            println("    - Casa a $pos (età: $(round(current_age_at_conversion, digits=1))) età cristallizzata.")
        end
        
        model.house_condition_noise[pos...] = 0.0 
    end
    println("  [Public Housing - Strategia Rent Gap] Processo di conversione completato.")
end


function get_vacant_private_houses_positions(model::ABM)
    vacant_private_houses = NTuple{2, Int}[]
    for x in 1:model.grid_dims[1], y in 1:model.grid_dims[2]
        pos = (x, y)
        # Una casa è "vacante e disponibile per il mercato privato" se:
        # 1. È di tipo :house
        # 2. È gestita privatamente (:private)
        # 3. Non è attualmente occupata
        if model.patch_type[x, y] == :house && model.management_type[x, y] == :private && isempty(ids_in_position(pos, model))
            push!(vacant_private_houses, pos)
        end
    end
    return vacant_private_houses
end


function get_vacant_public_housing_positions(model::ABM)
    vacant_ph_houses = NTuple{2, Int}[]
    for x in 1:model.grid_dims[1], y in 1:model.grid_dims[2]
        pos = (x, y)
        # MODIFICATO: Usa :public invece di :public_housing_managed
        if model.patch_type[x, y] == :house && model.management_type[x, y] == :public && isempty(ids_in_position(pos, model))
            push!(vacant_ph_houses, pos)
        end
    end
    return vacant_ph_houses
end