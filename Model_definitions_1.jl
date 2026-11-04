# === CELLA 2: DEFINIZIONI STRUCT E FUNZIONI HELPER ===

using Plots
using DataFrames 


function my_custom_random_scheduler(model::ABM)
    agent_ids = collect(allids(model)) # Prende tutti gli ID e li mette in un vettore
    return shuffle(model.rng, agent_ids) # Mescola gli ID usando il generatore casuale del modello
end

# --- Funzione Helper per parametri LogNormal ---
function calculate_lognormal_params(target_mean, target_variance)
    if target_mean <= 0 || target_variance <= 0
        error("Media e Varianza devono essere positive per LogNormal")
    end
    mean_sq = target_mean^2
    # Gestisce il caso var=0 -> sigma=0
    if target_variance == 0
        sigma_sq = 0.0
    else
        # Evita divisione per zero se mean_sq è zero (improbabile con check precedente)
        exp_sigma_sq = 1 + target_variance / (mean_sq + eps()) # Aggiunto eps() per stabilità se mean_sq è minuscolo
        if exp_sigma_sq <= 1 + eps() # Confronto con tolleranza
             error("Errore nel calcolo: exp(σ^2) <= 1 (target_variance potrebbe essere negativa o troppo piccola?)")
         end
         sigma_sq = log(exp_sigma_sq)
    end
    sigma = sqrt(sigma_sq)
    mu = log(target_mean + eps()) - sigma_sq / 2.0 # Aggiunto eps() per stabilità se target_mean è minuscolo
    return (μ=mu, σ=sigma)
end

# prima w_poor_for_dist = 0.65 e target_mean_poor = 20.0 e target_mean_rich = 65.0
function create_mixed_distribution(;w_poor_for_dist = 0.60, target_mean_poor = 30.0, target_var_poor = 10.0^2, target_mean_rich = 75.0, target_var_rich = 30.0^2)

    w_rich_for_dist = 1.0 - w_poor_for_dist
    weights_for_dist = [w_poor_for_dist, w_rich_for_dist]

    params_poor_ln = calculate_lognormal_params(target_mean_poor, target_var_poor)
    dist_poor_component = LogNormal(params_poor_ln.μ, params_poor_ln.σ) # Rinominato

    params_rich_ln = calculate_lognormal_params(target_mean_rich, target_var_rich)
    dist_rich_component = LogNormal(params_rich_ln.μ, params_rich_ln.σ) # Rinominato

    # ATTENZIONE AL NOME DELLA VARIABILE QUI SOTTO!
    # Evita 'components' se usi 'using Distributions' (import completo)
    distribution_component_list = [dist_poor_component, dist_rich_component]
    income_distribution_object = MixtureModel(distribution_component_list, weights_for_dist)
    return income_distribution_object

end

function sample_from_histogram(rng, bin_edges::Vector, probabilities::Vector; min_value = 7.0)
    # Assicurati che le lunghezze siano corrette
    @assert length(bin_edges) == length(probabilities) + 1 "Deve esserci un bordo in più rispetto alle probabilità."
    
    # Stage 1: Scegli l'indice di un bin in base alla sua probabilità
    bin_index = wsample(rng, 1:length(probabilities), probabilities)
    
    # Stage 2: Estrai un valore uniforme all'interno dei bordi di quel bin
    lower_bound = bin_edges[bin_index]
    upper_bound = bin_edges[bin_index + 1]
    
    # --- MODIFICA CHIAVE ---
    # Se abbiamo scelto il primo bin (che parte da 0), usiamo il nostro valore minimo.
    if bin_index == 1
        lower_bound = max(lower_bound, min_value)
    end
    
    return rand(rng) * (upper_bound - lower_bound) + lower_bound
end

function plot_actual_income_distribution(model::ABM; nbins_actual=50)
    # Estrai i redditi effettivi da tutti gli agenti nel modello
    actual_incomes = [a.income_value for a in allagents(model)]
    
    if isempty(actual_incomes)
        println("Nessun agente nel modello per plottare la distribuzione dei redditi.")
        return
    end

    # Recupera la distribuzione teorica dalle proprietà del modello
    # (Questa è stata passata e memorizzata durante initialize_model)
    theoretical_dist = model.income_distribution 
    pesi_teorici = probs(theoretical_dist.prior)
    distribuzioni_teoriche = theoretical_dist.components

    # Crea l'istogramma dei redditi effettivi
    p_actual_vs_theoretical = stephist(actual_incomes, bins=nbins_actual, normalize=:pdf, 
                                     label="Effettiva nel Modello (N=$(length(actual_incomes)))",
                                     xlabel="Valore Reddito", 
                                     ylabel="Densità",
                                     title="Distribuzione Redditi: Effettiva nel Modello vs. Teorica")

    # Determina un intervallo di valori per plottare le PDF teoriche
    # basandosi sui redditi effettivi per una buona visualizzazione
    min_val_actual = 0.0 # In genere i redditi non sono negativi
    max_val_actual = quantile(actual_incomes, 1.0) # Massimo reddito effettivo
    plot_range = range(min_val_actual, max_val_actual * 1.05, length=500) # Estendi un po' il range

    # Sovrapponi le PDF della distribuzione teorica
    plot!(p_actual_vs_theoretical, plot_range, x -> pesi_teorici[1] * pdf(distribuzioni_teoriche[1], x), 
          label="Teorica: Comp. Povera (PDF pesata)", lw=2, linestyle=:dash)
    plot!(p_actual_vs_theoretical, plot_range, x -> pesi_teorici[2] * pdf(distribuzioni_teoriche[2], x), 
          label="Teorica: Comp. Ricca (PDF pesata)", lw=2, linestyle=:dash)
    plot!(p_actual_vs_theoretical, plot_range, x -> pdf(theoretical_dist, x), 
          label="Teorica: PDF Totale Complessiva", lw=2, linestyle=:dot, color=:black)
    
    # Imposta i limiti dell'asse x in base ai redditi effettivi
    xlims!(p_actual_vs_theoretical, (min_val_actual, max_val_actual * 1.05))
    
    display(p_actual_vs_theoretical)
end


function plot_actual_income_distribution_gamma(model::ABM; nbins_actual=50)
    # Estrai i redditi effettivi da tutti gli agenti nel modello
    actual_incomes = [a.income_value for a in allagents(model)]
    
    if isempty(actual_incomes)
        println("Nessun agente nel modello per plottare la distribuzione dei redditi.")
        return
    end

    # Recupera la distribuzione teorica dalle proprietà del modello
    # (Questa è stata passata e memorizzata durante initialize_model)
    theoretical_dist = model.income_distribution 
    

    # Crea l'istogramma dei redditi effettivi
    p_actual_vs_theoretical = stephist(actual_incomes, bins=nbins_actual, normalize=:pdf, 
                                     label="Effettiva nel Modello (N=$(length(actual_incomes)))",
                                     xlabel="Valore Reddito", 
                                     ylabel="Densità",
                                     title="Distribuzione Redditi: Effettiva nel Modello vs. Teorica")

    # Determina un intervallo di valori per plottare le PDF teoriche
    # basandosi sui redditi effettivi per una buona visualizzazione
    min_val_actual = 0.0 # In genere i redditi non sono negativi
    max_val_actual = quantile(actual_incomes, 1.0) # Massimo reddito effettivo
    plot_range = range(min_val_actual, max_val_actual * 1.05, length=500) # Estendi un po' il range

    # Sovrapponi le PDF della distribuzione teorica

    plot!(p_actual_vs_theoretical, plot_range, x -> pdf(theoretical_dist, x), 
          label="Teorica: PDF Totale Complessiva", lw=2, linestyle=:dot, color=:black)
    
    # Imposta i limiti dell'asse x in base ai redditi effettivi
    xlims!(p_actual_vs_theoretical, (min_val_actual, max_val_actual * 1.05))
    
    percentile_poor = quantile(theoretical_dist, model.w_poor)
    println("Valore del 60° percentile: $(round(percentile_poor, digits=2))k \$")

    # Disegna una linea verticale sul grafico in corrispondenza di quel valore
    vline!(p_actual_vs_theoretical, 
        [percentile_poor], 
        label="60° Percentile", 
        color=:magenta, 
        linestyle=:dot, 
        linewidth=2
    )


    display(p_actual_vs_theoretical)
end

function plot_income_verification(model)
    # 1. Estrai i redditi effettivi degli agenti dal modello
    actual_incomes = [a.income_value for a in allagents(model)]
    if isempty(actual_incomes)
        println("Nessun agente nel modello per il plot.")
        return
    end

    # 2. Estrai i dati reali e la soglia direttamente dalle proprietà del modello
    income_cutoff = model.income_cutoff
    real_midpoints = model.real_income_midpoints
    real_weights = model.real_income_weights
    real_bins = model.real_income_bins

    # 3. Crea il grafico di base con l'istogramma dei dati reali
    p_verification = stephist(real_midpoints, weights=real_weights, normalize=:pdf,
        bins=real_bins,
        label="Empirical distribution (USA 2013)",
        #title="Household income distribution",
        xlabel="Annual income (thousands of \$)",
        ylabel="Probability density",
        color=:dodgerblue,
        linestyle=:dash,
        linewidth=2.5,
        legend = (0.5, 0.75),
        # --- MODIFICHE PER QUALITÀ E LEGENDA ---
        #size=(1000, 750),      # Aumenta le dimensioni del plot in pixel (larghezza, altezza)
        dpi=300,               # Aumenta la risoluzione (dots per inch) per una qualità superiore
        legendfontsize=10,
        titlefontsize=16,      # (Opzionale) Aumenta la dimensione del titolo
        guidefontsize=12
    )

    # 4. Sovrapponi l'istogramma dei redditi degli agenti creati
    Plots.stephist!(p_verification, actual_incomes,
        bins=real_bins,
        normalize=:pdf,
        label="Simulated distribution (N=$(length(actual_incomes)))",
        linewidth=1.5,
        color=:gray
    )

    # 5. Disegna la linea della soglia percentile
    vline!(p_verification, [income_cutoff],
        label="$(round(Int, model.w_poor_percentile*100))° Percentile threshold",
        color=:magenta,
        linestyle=:dot,
        linewidth=2.5
    )
    
    display(p_verification)
end


using Plots, StatsPlots, Statistics

function plot_rent_burden_distribution(model; real_midpoints, real_weights, real_bins, current_seed)
    
    # --- Calcola i rapporti affitto/reddito per ogni agente ---
    ratios = Float64[]
    for agent in allagents(model)
        annual_income_k = agent.income_value 
        if annual_income_k > 0
            annual_rent_k = model.capitalized_rent[agent.pos...]
            
            # MODIFICA: Calcolo diretto tra valori annuali
            push!(ratios, (annual_rent_k / annual_income_k) * 100)
        end
    end

    if isempty(ratios); println("Nessun agente trovato."); return; end


    # --- NUOVO: Calcola e stampa la mediana ---
    median_ratio_model = median(ratios)
    println("Mediana del carico d'affitto nel modello: $(round(median_ratio_model, digits=1))%")


    # --- Creazione del Grafico di Confronto ---

    # 1. Plotta l'istogramma dei DATI REALI
    p = stephist(real_midpoints, weights=real_weights, normalize=:pdf,
        bins=real_bins,
        label="Empirical distribution (USA 2013)",
        #title="Distribuzione Carico d'Affitto: Modello vs. Reale",
        xlabel="Rent as % of the Income",
        ylabel="Probability density",
        color=:gray, linestyle=:dash, linewidth=2.5
    )

    # 2. Sovrapponi l'istogramma dei dati del MODELLO
    Plots.stephist!(p, ratios,
        bins=real_bins,
        normalize=:pdf,
        label="Simulated distribution (seed = $current_seed)",
        linewidth=1.5,
        color=:dodgerblue
    )

    # Aggiungi una linea verticale per la mediana del modello
    #vline!(p, [median_ratio_model], label="Mediana Modello", color=:crimson, linestyle=:dot, lw=2)


    display(p)
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


# Funzioni helper definite precedentemente (get_low_income_color_from_norm, get_high_income_color_from_norm)
# ... (assicurati che siano definite prima di city_plot o all'interno se preferisci)



function calculate_financial_utility(agent::AgentType, house_pos::NTuple{2, Int}, model::ABM)
    
    income = agent.income_value
    rent_matrix = model.capitalized_rent
    rent = rent_matrix[house_pos...]

    # Caso 1: Reddito nullo o negativo
    if income <= 0.0
        return rent > 0.0 ? 0.0 : 1.0 # Utilità 1.0 solo se anche l'affitto è <= 0.0
    end

    # Caso 2: Affitto nullo o negativo (con reddito positivo, grazie al check precedente)
    if rent <= 0.0
        return 1.0
    end

    # Caso 3: Affitto uguale o superiore al reddito (con reddito positivo)
    if rent >= income
        return 0.0
    end

    # Caso standard: 0 < rent < income
    # L'utilità è la frazione del reddito rimanente.
    # (income - rent) / income  è già tra (0, 1) esclusi estremi.
    financial_utility = (income - rent) / income
    
    # Per maggiore sicurezza, anche se la logica dovrebbe già garantirlo per il caso standard.
    return clamp(financial_utility, 0.0, 1.0)
end


function calculate_amenity_utility(house_pos::NTuple{2, Int}, model::ABM)
    
    current_raw_score = model.raw_amenity_scores_matrix[house_pos...]

    # Gestione del caso in cui global_amenity_scores potrebbe essere vuoto
    if isnan(current_raw_score)
        # Potresti voler stampare un avviso qui se non ti aspetti che accada
        println("Attenzione: punteggio amenità grezzo NaN per la posizione $house_pos")
        return 0.0
    end

    # Usa nomi di variabili diversi per evitare shadowing delle funzioni base
    overall_min_score = model.global_min_raw_amenity_score
    overall_max_score = model.global_max_raw_amenity_score

    # Normalizzazione Min-Max con gestione della divisione per zero
    local normalized_amenity_utility::Float64
    range_amenity = overall_max_score - overall_min_score

    if abs(range_amenity) < 1e-9 # Se il range globale è (quasi) zero
        # Se tutti i punteggi sono uguali (e non NaN), normalizza a 0.5 
        # (o 0.0 se il punteggio stesso è 0, anche se questo caso è coperto da clamp)
        # Se il punteggio è l'unico valore, allora (score - score) / (score - score) è problematico.
        # Se min == max, allora tutti i punteggi normalizzati dovrebbero essere uguali.
        # Se quel valore è 0, tutti 0. Altrimenti, tutti 0.5 o 1 a seconda della convenzione.
        # Qui, se tutti i punteggi sono, ad esempio, 5, e current_raw_score è 5,
        # (5 - 5) / (5 - 5) -> 0/0.
        # Se il range è zero, significa che tutti i punteggi sono uguali al minimo (e al massimo).
        # Quindi tutti dovrebbero avere la stessa utilità normalizzata. 0.5 è una scelta comune.
        # Se il valore comune è 0, l'utilità dovrebbe essere 0.
        normalized_amenity_utility = (abs(overall_min_score) < 1e-9 && abs(current_raw_score) < 1e-9) ? 0.0 : 0.5
    else
        normalized_amenity_utility = (current_raw_score - overall_min_score) / range_amenity
    end
    
    # Clampa il risultato per assicurare che sia strettamente in [0,1]
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
    
    # 1. Ottieni l'affitto della casa e gestisci se è NaN
    # rent_val è necessario per la condizione dell'hard gate.
    # calculate_financial_utility accede a model.capitalized_rent[house_pos...] internamente.
    rent_val = model.capitalized_rent[house_pos...]
    if isnan(rent_val)
        # Se l'affitto è NaN, questa casa è problematica, l'utilità dovrebbe essere 0.
        # Questo potrebbe succedere se house_pos non è una casa valida o completamente inizializzata.
        # println("Attenzione: affitto NaN per $house_pos in calculate_total_utility")
        return 0.0
    end

    # 2. Calcola l'utilità finanziaria
    u_fin = calculate_financial_utility(agent, house_pos, model)

    # 3. "Hard Gate": se u_fin è 0 E la casa ha un costo positivo, l'utilità totale è 0.
    # Questo significa che una casa inaccessibile che costa qualcosa non darà utilità.
    # Se rent_val <= 0 (gratis/sussidio), u_fin sarà 1.0, quindi questa condizione non scatta.
    if u_fin == 0.0 && rent_val > 0.0
        return 0.0
    end

    # 4. Se siamo qui, la casa è o finanziariamente sostenibile (u_fin > 0),
    #    o esattamente al limite (u_fin = 0 ma rent_val = 0, caso raro e gestito da u_fin=1),
    #    o è gratuita/sussidiata (u_fin = 1.0).
    #    Procedi a calcolare le altre componenti e la somma pesata.

    age_matrix = model.house_age
    u_amenity = calculate_amenity_utility(house_pos, model) # Assumiamo house_pos sia valida

    # Calcolo robusto di u_age (rinominato in u_age_calculated per chiarezza)
    current_house_age = age_matrix[house_pos...] 
    u_age_calculated = 0.0 # Valore di default per u_age
    if !isnan(current_house_age) # Verifica se l'età è un numero valido
        if model.current_max_observed_age > 0.0 # Procedi solo se max_house_age è positivo
            raw_u_age = (model.current_max_observed_age - current_house_age) / model.current_max_observed_age
            u_age_calculated = clamp(raw_u_age, 0.0, 1.0)
        elseif model.current_max_observed_age == 0.0 && current_house_age == 0.0 
            # Caso speciale: nessuna età massima definita (o 0) E la casa è nuova
            u_age_calculated = 1.0 # Utilità di età massima
        # In altri casi (es. max_house_age è 0 ma current_house_age non è 0, o current_house_age è NaN),
        # u_age_calculated rimane 0.0, che è una gestione sicura.
        end
    end

    if u_age_calculated < 0.05
        return 0.0
    end
    


    dist_from_center = sqrt((house_pos[1] - model.cbd_center[1])^2 + (house_pos[2] - model.cbd_center[2])^2)
    u_center = clamp(1.0 - dist_from_center / model.max_dist_from_center_to_corner, 0.0, 1.0)

    # --- NUOVA PARTE ---
    # Calcola la nuova utilità di vicinato
    u_neighborhood = calculate_neighborhood_utility(agent, house_pos, model)
    # ------------------

    local weights
    if agent isa LowIncome
        weights = model.utility_weights_low 
    else # HighIncome
        weights = model.utility_weights_high
    end

    # Aggiorna la somma pesata per includere la nuova componente
    total_u_raw = (weights.financial * u_fin +
                   weights.amenity * u_amenity +
                   weights.age * u_age_calculated +
                   weights.center * u_center +
                   weights.neighborhood * u_neighborhood) # <-- AGGIUNTA
                      
    return clamp(total_u_raw, 0.0, 1.0)


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
            Makie.Colorbar(fig_abm[2, 1], colormap = low_income_cb_colors, limits = (min_inc_actual, max_inc_actual), label = "Reddito (LowIncome)", vertical = false, height = 20, flipaxis = false )
        end
        if isdefined(Main, :get_high_income_color_from_norm)
            high_income_cb_colors = [get_high_income_color_from_norm(n) for n in range(0, 1, length=num_cb_samples)]
            Makie.Colorbar(fig_abm[3, 1], colormap = high_income_cb_colors, limits = (min_inc_actual, max_inc_actual), label = "Reddito (HighIncome)", vertical = false, height = 20, flipaxis = false )
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
            low_income_cb_colors = [get_low_income_color_from_norm(norm_inc) for norm_inc in range(0, 1, length=num_colorbar_samples)]
            Makie.Colorbar(fig_abm[2, 1], colormap = low_income_cb_colors, limits = (0, 65), label = "Reddito (LowIncome Agenti)", vertical = false, height = 20, flipaxis = false )
        end
        if isdefined(Main, :get_high_income_color_from_norm)
            high_income_cb_colors = [get_high_income_color_from_norm(norm_inc) for norm_inc in range(0, 1, length=num_colorbar_samples)]
            Makie.Colorbar(fig_abm[3, 1], colormap = high_income_cb_colors, limits = (65, 300), label = "Reddito (HighIncome Agenti)", vertical = false, height = 20, flipaxis = false )
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
                axis = (; title = "Affitto Potenziale e Caratteristiche Patch"),
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
        cb_potential_rent = Colorbar(fig_pot_rent_detailed[1, 2], 
            colormap = potential_rent_sub_colormap, 
            limits = (min_pot_rent_plot, max_pot_rent_plot),
            label = "Affitto Potenziale (Case)", width = 20)

        cb_amenity_level = Colorbar(fig_pot_rent_detailed[1, 3], 
            colormap = amenity_greens_sub_colormap, 
            limits = (min_amenity_plot, max_amenity_plot), 
            label = "Livello Amenità (Aree Pubbliche)", width = 20)
        
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


println("Cella 2: Struct agenti e funzioni helper definite.")


# Preferibilmente in CELLA 2 con le altre definizioni di funzioni,
# o all'inizio di CELLA 4 prima del suo utilizzo.
# Assicurati che Plots e StatsPlots siano stati importati (es. using Plots, StatsPlots)



# Assicurati che Plots e StatsPlots siano importati
# using Plots
# using StatsPlots # Necessario per histogram con raggruppamento e normalize=:pdf

using Plots
using StatsPlots # Necessario per histogram con raggruppamento e normalize=:pdf

function plot_satisfaction_distributions(model::ABM; nbins=20,  plot_title="Distribuzione Soddisfazione per Classe")
    # Rimosso: using Plots (deve essere caricato a livello globale con `using Plots`)
    
    low_income_utilities = Float64[]
    high_income_utilities = Float64[]

    if nagents(model) == 0
        println("Nessun agente nel modello per calcolare la soddisfazione.")
        return
    end

    for agent in allagents(model)
        current_utility = calculate_total_utility(agent, agent.pos, model)
        if agent isa LowIncome
            push!(low_income_utilities, current_utility)
        else # agent isa HighIncome
            push!(high_income_utilities, current_utility)
        end
    end

    all_utilities = vcat(low_income_utilities, high_income_utilities)
    groups = vcat(fill("LowIncome (N=$(length(low_income_utilities)))", length(low_income_utilities)), 
                  fill("HighIncome (N=$(length(high_income_utilities)))", length(high_income_utilities)))

    if isempty(all_utilities)
        println("Nessuna utilità calcolata (forse nessun agente?).")
        return
    end
    
    println("\n--- Statistiche Soddisfazione Iniziale ---")
    if !isempty(low_income_utilities)
        println("LowIncome: Media U = $(mean(low_income_utilities)), Mediana U = $(median(low_income_utilities)), Min U = $(minimum(low_income_utilities)), Max U = $(maximum(low_income_utilities))")
    end
    if !isempty(high_income_utilities)
        println("HighIncome: Media U = $(mean(high_income_utilities)), Mediana U = $(median(high_income_utilities)), Min U = $(minimum(high_income_utilities)), Max U = $(maximum(high_income_utilities))")
    end
    
    plt_satisfaction = Plots.histogram(all_utilities, # <--- MODIFICATO: Plots.histogram
                                 group = groups, 
                                 bins = nbins, 
                                 normalize = :pdf,
                                 title = plot_title,
                                 xlabel = "Utilità Totale (0-1)",
                                 ylabel = "Densità",
                                 legend = :topright,
                                 opacity = 0.7)

    if !isempty(low_income_utilities)
        mean_low = mean(low_income_utilities)
        Plots.vline!(plt_satisfaction, [mean_low], linestyle=:dash, color=palette(:default)[1], # <--- MODIFICATO: Plots.vline!
               label="", lw=2) 
    end
    if !isempty(high_income_utilities)
        mean_high = mean(high_income_utilities)
        Plots.vline!(plt_satisfaction, [mean_high], linestyle=:dash, color=palette(:default)[2], # <--- MODIFICATO: Plots.vline!
               label="", lw=2) 
    end

    Plots.xlims!(plt_satisfaction, (-0.05, 1.05)) # <--- MODIFICATO: Plots.xlims!
    Plots.ylims!(plt_satisfaction, (0, Inf))    # <--- MODIFICATO: Plots.ylims!

    display(plt_satisfaction)
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



# Assicurati che i pacchetti necessari siano importati:
# using Agents, Distributions, Random, Statistics, DataFrames



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

function process_simulation_results(agent_df::DataFrame)
    if isempty(agent_df)
        println("DataFrame dei dati grezzi è vuoto, impossibile processare i risultati.")
        # Restituisce un DataFrame vuoto con la struttura attesa se possibile,
        # o semplicemente un DataFrame vuoto generico.
        return DataFrame(step=Int[], agent_type=Symbol[], mean_utility=Float64[], std_utility=Float64[], count=Int[], sem_utility=Float64[])
    end
    
    println("Processamento dei dati della simulazione per creare il riepilogo...")
    
    # Raggruppa per step e tipo di agente, poi calcola media e deviazione standard dell'utilità
    summary_df = combine(groupby(agent_df, [:step, :agent_type]), 
                         :utility => mean => :mean_utility,
                         :utility => std => :std_utility,
                         nrow => :count  # Numero di agenti in quel gruppo/step
                        )
    
    # Calcola l'errore standard della media (SEM = std / sqrt(N))
    # Gestisci il caso in cui std_utility potrebbe essere NaN (se count è 1, std è NaN)
    # o se count è 0 (anche se groupby non dovrebbe produrre gruppi vuoti).
    summary_df.sem_utility = map(zip(summary_df.std_utility, summary_df.count)) do (sd, n)
        if n > 0
            # Se la deviazione standard è NaN (tipicamente se n=1), SEM è 0 o NaN.
            # Per il plotting, 0 è spesso più gestibile di NaN per ribbon.
            # Se n=1, la "media" è solo quel singolo valore, e l'incertezza sulla media è difficile da definire.
            # Potremmo dire che SEM è 0 se n=1.
            n == 1 ? 0.0 : (isnan(sd) ? 0.0 : sd / sqrt(n))
        else
            0.0 # Nessun agente, nessun errore standard
        end
    end
    
    println("Processamento dati completato.")
    return summary_df
end


# Assicurati che i pacchetti necessari siano importati:
# using Plots
# using StatsPlots # Per @df

function plot_satisfaction_trend_from_summary(
    summary_df_to_plot::DataFrame; 
    plot_title::String = "Andamento Soddisfazione Media Agenti", 
    y_axis_limits::Tuple{Float64, Float64} = (0.2,0.7),
    ribbon_column_name::Symbol = :sem_utility )
    if isempty(summary_df_to_plot)
        println("DataFrame di riepilogo vuoto, impossibile generare il plot.")
        return nothing 
    end

    # Verifica che le colonne necessarie esistano
    required_cols = [:step, :mean_utility, :agent_type, ribbon_column_name]
    missing_cols = setdiff(required_cols, Symbol.(names(summary_df_to_plot)))
    if !isempty(missing_cols)
        println("Colonne mancanti nel DataFrame di riepilogo: $missing_cols. Impossibile generare il plot.")
        println("Colonne disponibili: $(names(summary_df_to_plot))")
        return nothing
    end

    println("Creazione del grafico dell'andamento della soddisfazione media (approccio iterativo)...")
    
    # Inizializza un plot vuoto
    # Qualifica Plots.plot se hai conflitti di nomi (es. con AgentsPlots o CairoMakie)
    plt_trend = Plots.plot(
        legend = :bottomright,
        title = plot_title,
        xlabel = "Step di Simulazione",
        ylabel = "Soddisfazione Media",
        linewidth = 2
    )

    # Itera su ciascun tipo di agente (gruppo)
    grouped_data = groupby(summary_df_to_plot, :agent_type)
    
    for group_df in grouped_data
        agent_type_label = string(first(group_df.agent_type)) # Etichetta per la legenda
        
        # Estrai i vettori di dati per questo gruppo
        steps = group_df.step
        mean_utilities = group_df.mean_utility
        ribbon_values = group_df[!, ribbon_column_name] # Accedi alla colonna del ribbon per nome

        # Aggiungi la serie al plot, passando esplicitamente il vettore per il ribbon
        Plots.plot!(plt_trend, steps, mean_utilities, 
                    label = agent_type_label, 
                    ribbon = ribbon_values)
    end
    
    Plots.ylims!(plt_trend, y_axis_limits) 
    
    display(plt_trend) 
    return plt_trend   
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

function calculate_all_city_rent_metrics(model::ABM)
    all_city_rents = Float64[]

    # Itera su tutte le posizioni della griglia
    for r_idx in 1:model.grid_dims[1], c_idx in 1:model.grid_dims[2]
        current_pos = (r_idx, c_idx)
        
        # Controlla se la patch è una casa privata o public housing
        current_patch_type = model.patch_type[current_pos...]
        if (current_patch_type == :house || current_patch_type == :public_housing)
            rent_val = model.capitalized_rent[current_pos...]
            if !isnan(rent_val)
                push!(all_city_rents, rent_val)
            end
        end
    end

    # Calcola sia la media che la mediana
    mean_rent = !isempty(all_city_rents) ? mean(all_city_rents) : NaN
    median_rent = !isempty(all_city_rents) ? median(all_city_rents) : NaN

    return (
        mean_city_rent = mean_rent,
        median_city_rent = median_rent,
        n_houses_city_rent_calc = length(all_city_rents)
    )
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


function plot_lh_ratio_trend(data_df::DataFrame; 
                             title_suffix::String="", 
                             y_max_ratio::Union{Float64, Symbol}=:auto)
    if isempty(data_df)
        println("DataFrame per LH Ratio trend vuoto.")
        return nothing
    end

    # Determina se i dati sono aggregati (multi-run) o da singola run
    is_aggregated = :mean_ratio_lh_inner in Symbol.(names(data_df))
    
    col_inner = is_aggregated ? :mean_ratio_lh_inner : :ratio_lh_inner
    col_suburbs = is_aggregated ? :mean_ratio_lh_suburbs : :ratio_lh_suburbs
    sem_inner = is_aggregated ? ( :sem_ratio_lh_inner in Symbol.(names(data_df)) ? data_df.sem_ratio_lh_inner : nothing ) : nothing
    sem_suburbs = is_aggregated ? ( :sem_ratio_lh_suburbs in Symbol.(names(data_df)) ? data_df.sem_ratio_lh_suburbs : nothing ) : nothing

    if !(col_inner in Symbol.(names(data_df))) || !(col_suburbs in Symbol.(names(data_df)))
        println("Colonne necessarie per LH Ratio trend mancanti: $col_inner, $col_suburbs")
        return nothing
    end

    plt = Plots.plot(legend=:outertopright, 
                     title="Evoluzione Rapporto Low/High Income" * title_suffix,
                     xlabel="Step di Simulazione", 
                     ylabel="Rapporto LowIncome / HighIncome")

    Plots.plot!(plt, data_df.step, data_df[!, col_inner],
        label = "Inner City", ribbon = sem_inner, 
        linewidth = 2, seriesalpha = 0.8)

    Plots.plot!(plt, data_df.step, data_df[!, col_suburbs],
        label = "Suburbs", ribbon = sem_suburbs, 
        linewidth = 2, seriesalpha = 0.8)

    current_ylims = Plots.ylims(plt)
    final_y_max = y_max_ratio == :auto ? current_ylims[2] : (isa(y_max_ratio, Real) && !isnan(y_max_ratio) ? y_max_ratio : current_ylims[2])
    final_y_min = max(0.0, current_ylims[1])
    
    all_inner_nan = all(isnan, data_df[!, col_inner])
    all_suburbs_nan = all(isnan, data_df[!, col_suburbs])

    if all_inner_nan && all_suburbs_nan
         Plots.ylims!(plt, (0, 1.0)) 
    elseif final_y_max > final_y_min 
         Plots.ylims!(plt, (final_y_min, final_y_max))
    end

    display(plt)
    return plt
end

"""
    plot_total_density_trend(summary_df_model_metrics::DataFrame; title_suffix="")
Plotta l'andamento temporale della densità totale di agenti per zona.
`summary_df_model_metrics` deve contenere :step, :mean_density_total_inner, :sem_density_total_inner, ecc.
"""
function plot_total_density_trend(data_df::DataFrame; title_suffix::String="")
    if isempty(data_df)
        println("DataFrame per Total Density trend vuoto.")
        return nothing
    end

    is_aggregated = :mean_density_total_inner in Symbol.(names(data_df))

    col_inner = is_aggregated ? :mean_density_total_inner : :density_total_inner
    col_suburbs = is_aggregated ? :mean_density_total_suburbs : :density_total_suburbs
    sem_inner = is_aggregated ? ( :sem_density_total_inner in Symbol.(names(data_df)) ? data_df.sem_density_total_inner : nothing ) : nothing
    sem_suburbs = is_aggregated ? ( :sem_density_total_suburbs in Symbol.(names(data_df)) ? data_df.sem_density_total_suburbs : nothing ) : nothing

    if !(col_inner in Symbol.(names(data_df))) || !(col_suburbs in Symbol.(names(data_df)))
        println("Colonne necessarie per Total Density trend mancanti: $col_inner, $col_suburbs")
        return nothing
    end

    plt = Plots.plot(legend=:outertopright, # <--- Già corretto qui (Plots.plot)
                      title="Evoluzione Densità Totale Agenti" * title_suffix,
                      xlabel="Step di Simulazione",
                      ylabel="Densità Agenti (per patch house)")

    Plots.plot!(plt, data_df.step, data_df[!, col_inner],
        label = "Inner City", ribbon = sem_inner,
        linewidth = 2, seriesalpha = 0.8)

    Plots.plot!(plt, data_df.step, data_df[!, col_suburbs],
        label = "Suburbs", ribbon = sem_suburbs,
        linewidth = 2, seriesalpha = 0.8)

    max_obs_density = 0.0
    if !all(isnan, data_df[!, col_inner]); max_obs_density = max(max_obs_density, maximum(filter(!isnan, data_df[!, col_inner]))); end
    if !all(isnan, data_df[!, col_suburbs]); max_obs_density = max(max_obs_density, maximum(filter(!isnan, data_df[!, col_suburbs]))); end
    y_upper_limit = max(1.0, max_obs_density * 1.1)

    Plots.ylims!(plt, (0, y_upper_limit)) # <--- MODIFICATO: Aggiunto Plots.
    display(plt)
    return plt
end

function plot_absolute_agent_counts_trend(data_df::DataFrame; 
                                           zone_to_plot::Symbol, 
                                           title_suffix::String="")
    if isempty(data_df); println("DataFrame per Agent Counts trend vuoto."); return nothing; end
    
    zone_prefix = zone_to_plot == :inner_city ? "inner" : "suburbs"
    plot_zone_label = zone_to_plot == :inner_city ? "Inner City" : "Suburbs"

    is_aggregated = Symbol("mean_n_low_", zone_prefix) in Symbol.(names(data_df))

    col_low = is_aggregated ? Symbol("mean_n_low_", zone_prefix) : Symbol("n_low_", zone_prefix)
    col_high = is_aggregated ? Symbol("mean_n_high_", zone_prefix) : Symbol("n_high_", zone_prefix)
    sem_low = is_aggregated ? (Symbol("sem_n_low_", zone_prefix) in Symbol.(names(data_df)) ? data_df[!, Symbol("sem_n_low_", zone_prefix)] : nothing) : nothing
    sem_high = is_aggregated ? (Symbol("sem_n_high_", zone_prefix) in Symbol.(names(data_df)) ? data_df[!, Symbol("sem_n_high_", zone_prefix)] : nothing) : nothing

    if !(col_low in Symbol.(names(data_df))) || !(col_high in Symbol.(names(data_df)))
        println("Colonne necessarie per Agent Counts trend mancanti per zona '$zone_prefix': $col_low, $col_high")
        return nothing
    end

    plt = Plots.plot(legend=:outertopright, 
                     title="Evoluzione Numero Agenti in $plot_zone_label" * title_suffix,
                     xlabel="Step di Simulazione", 
                     ylabel="Numero Medio Agenti")

    # LowIncome Agents
    Plots.plot!(plt, data_df.step, data_df[!, col_low],
        label = "LowIncome in $plot_zone_label", ribbon = sem_low, 
        linewidth = 2, seriesalpha = 0.8, color = :blue)

    # HighIncome Agents
    Plots.plot!(plt, data_df.step, data_df[!, col_high],
        label = "HighIncome in $plot_zone_label", ribbon = sem_high, 
        linewidth = 2, seriesalpha = 0.8, color = :red)

    max_obs_count = 0.0
    if !all(isnan, data_df[!, col_low]); max_obs_count = max(max_obs_count, maximum(filter(!isnan, data_df[!, col_low]))); end
    if !all(isnan, data_df[!, col_high]); max_obs_count = max(max_obs_count, maximum(filter(!isnan, data_df[!, col_high]))); end
    y_upper_limit = max(10.0, max_obs_count * 1.1)

    Plots.ylims!(plt, (0, y_upper_limit)) # <--- MODIFICATO
    display(plt)
    return plt
end


function plot_segregation_trend(data_df::DataFrame; title_suffix::String="")
    if isempty(data_df)
        println("DataFrame per il trend di segregazione è vuoto. Impossibile generare il plot.")
        return nothing
    end

    is_aggregated = :mean_neighbour_segregation_index in Symbol.(names(data_df))

    col_neighbour = is_aggregated ? :mean_neighbour_segregation_index : :neighbour_segregation_index
    col_amenity_low = is_aggregated ? :mean_amenity_segregation_index_low : :amenity_segregation_index_low
    col_amenity_high = is_aggregated ? :mean_amenity_segregation_index_high : :amenity_segregation_index_high

    sem_neighbour = is_aggregated && (:sem_neighbour_segregation_index in Symbol.(names(data_df))) ? data_df.sem_neighbour_segregation_index : nothing
    sem_amenity_low = is_aggregated && (:sem_amenity_segregation_index_low in Symbol.(names(data_df))) ? data_df.sem_amenity_segregation_index_low : nothing
    sem_amenity_high = is_aggregated && (:sem_amenity_segregation_index_high in Symbol.(names(data_df))) ? data_df.sem_amenity_segregation_index_high : nothing

    required_cols = [col_neighbour, col_amenity_low, col_amenity_high]
    for col in required_cols
        if !(col in Symbol.(names(data_df)))
            println("Errore: Colonna '$col' necessaria per il trend di segregazione non trovata nel DataFrame.")
            return nothing
        end
    end

    all_data_for_ylims = Float64[]

    if !isnothing(data_df[!, col_neighbour])
        append!(all_data_for_ylims, skipmissing(data_df[!, col_neighbour]))
        if !isnothing(sem_neighbour)
            append!(all_data_for_ylims, skipmissing(data_df[!, col_neighbour] .- sem_neighbour))
            append!(all_data_for_ylims, skipmissing(data_df[!, col_neighbour] .+ sem_neighbour))
        end
    end

    if !isnothing(data_df[!, col_amenity_low])
        append!(all_data_for_ylims, skipmissing(data_df[!, col_amenity_low]))
        if !isnothing(sem_amenity_low)
            append!(all_data_for_ylims, skipmissing(data_df[!, col_amenity_low] .- sem_amenity_low))
            append!(all_data_for_ylims, skipmissing(data_df[!, col_amenity_low] .+ sem_amenity_low))
        end
    end

    if !isnothing(data_df[!, col_amenity_high])
        append!(all_data_for_ylims, skipmissing(data_df[!, col_amenity_high]))
        if !isnothing(sem_amenity_high)
            append!(all_data_for_ylims, skipmissing(data_df[!, col_amenity_high] .- sem_amenity_high))
            append!(all_data_for_ylims, skipmissing(data_df[!, col_amenity_high] .+ sem_amenity_high))
        end
    end
    
    if isempty(all_data_for_ylims) || all(isnan, all_data_for_ylims)
        y_lower_bound = 0.0
        y_upper_bound = 1.05
    else
        min_val = minimum(skipnan(all_data_for_ylims))
        max_val = maximum(skipnan(all_data_for_ylims))

        y_lower_bound = max(0.0, min_val * 0.98 - 0.01)
        y_upper_bound = min(1.0, max_val * 1.02 + 0.01)

        if y_upper_bound - y_lower_bound < 0.05
            center = (y_lower_bound + y_upper_bound) / 2
            y_lower_bound = max(0.0, center - 0.025)
            y_upper_bound = min(1.0, center + 0.025)
        end
    end
    
    if (y_upper_bound - y_lower_bound) < 0.1 && (y_upper_bound <= 1.0 || y_lower_bound >= 0.0)
        current_range_center = (y_lower_bound + y_upper_bound) / 2
        y_lower_bound = max(0.0, current_range_center - 0.05)
        y_upper_bound = min(1.0, current_range_center + 0.05)
    end

    y_axis_limits_segregation = (y_lower_bound, y_upper_bound)

    println("Limiti Y calcolati per la segregazione: $y_axis_limits_segregation")

    plot_title = is_aggregated ? "Evoluzione Media Indici di Segregazione" : "Evoluzione Indici di Segregazione"
    y_label = is_aggregated ? "Indice di Segregazione (Media ± SEM)" : "Indice di Segregazione"

    plt = Plots.plot( # <--- MODIFICATO: Plots.plot
        legend=:bottomright,
        title=plot_title * title_suffix,
        xlabel="Step di Simulazione",
        ylabel=y_label,
        ylims=y_axis_limits_segregation,
        size=(1000, 600)
    )

    Plots.plot!(plt, data_df.step, data_df[!, col_neighbour],
        label = is_aggregated ? "Segregazione (Neighbour) Media" : "Segregazione (Neighbour)",
        linewidth = 2,
        seriesalpha = 0.8,
        color = :purple,
        line = (:solid),
        ribbon = sem_neighbour,
        fillalpha = 0.1
    )

    Plots.plot!(plt, data_df.step, data_df[!, col_amenity_low],
        label = is_aggregated ? "Esposizione Amenità (LowIncome) Media" : "Esposizione Amenità (LowIncome)",
        linewidth = 2,
        seriesalpha = 0.8,
        color = :red,
        line = (:dash),
        ribbon = sem_amenity_low,
        fillalpha = 0.1
    )

    Plots.plot!(plt, data_df.step, data_df[!, col_amenity_high],
        label = is_aggregated ? "Esposizione Amenità (HighIncome) Media" : "Esposizione Amenità (HighIncome)",
        linewidth = 2,
        seriesalpha = 0.8,
        color = :blue,
        line = (:dot),
        ribbon = sem_amenity_high,
        fillalpha = 0.1
    )
    
    display(plt)
    return plt
end

function plot_segregation_trend_bis(data_df::DataFrame; title_suffix::String="")
    if isempty(data_df)
        println("DataFrame per il trend di segregazione è vuoto. Impossibile generare il plot.")
        return nothing
    end

    is_aggregated = :mean_neighbour_segregation_index in Symbol.(names(data_df))

    col_neighbour = is_aggregated ? :mean_neighbour_segregation_index : :neighbour_segregation_index
    col_amenity_low = is_aggregated ? :mean_amenity_segregation_index_low : :amenity_segregation_index_low
    col_amenity_high = is_aggregated ? :mean_amenity_segregation_index_high : :amenity_segregation_index_high

    sem_neighbour = is_aggregated && (:sem_neighbour_segregation_index in Symbol.(names(data_df))) ? data_df.sem_neighbour_segregation_index : nothing
    sem_amenity_low = is_aggregated && (:sem_amenity_segregation_index_low in Symbol.(names(data_df))) ? data_df.sem_amenity_segregation_index_low : nothing
    sem_amenity_high = is_aggregated && (:sem_amenity_segregation_index_high in Symbol.(names(data_df))) ? data_df.sem_amenity_segregation_index_high : nothing

    required_cols = [col_neighbour, col_amenity_low, col_amenity_high]
    for col in required_cols
        if !(col in Symbol.(names(data_df)))
            println("Errore: Colonna '$col' necessaria per il trend di segregazione non trovata nel DataFrame.")
            return nothing
        end
    end

    all_data_for_ylims = Float64[]

    if !isnothing(data_df[!, col_neighbour])
        append!(all_data_for_ylims, skipmissing(data_df[!, col_neighbour]))
        if !isnothing(sem_neighbour)
            append!(all_data_for_ylims, skipmissing(data_df[!, col_neighbour] .- sem_neighbour))
            append!(all_data_for_ylims, skipmissing(data_df[!, col_neighbour] .+ sem_neighbour))
        end
    end

    if !isnothing(data_df[!, col_amenity_low])
        append!(all_data_for_ylims, skipmissing(data_df[!, col_amenity_low]))
        if !isnothing(sem_amenity_low)
            append!(all_data_for_ylims, skipmissing(data_df[!, col_amenity_low] .- sem_amenity_low))
            append!(all_data_for_ylims, skipmissing(data_df[!, col_amenity_low] .+ sem_amenity_low))
        end
    end

    if !isnothing(data_df[!, col_amenity_high])
        append!(all_data_for_ylims, skipmissing(data_df[!, col_amenity_high]))
        if !isnothing(sem_amenity_high)
            append!(all_data_for_ylims, skipmissing(data_df[!, col_amenity_high] .- sem_amenity_high))
            append!(all_data_for_ylims, skipmissing(data_df[!, col_amenity_high] .+ sem_amenity_high))
        end
    end
    
    if isempty(all_data_for_ylims) || all(isnan, all_data_for_ylims)
        y_lower_bound = 0.5
        y_upper_bound = 0.8
    else
        min_val = minimum(skipnan(all_data_for_ylims))
        max_val = maximum(skipnan(all_data_for_ylims))

        y_lower_bound = max(0.0, min_val * 0.98 - 0.01)
        y_upper_bound = min(1.0, max_val * 1.02 + 0.01)

        if y_upper_bound - y_lower_bound < 0.05
            center = (y_lower_bound + y_upper_bound) / 2
            y_lower_bound = max(0.0, center - 0.025)
            y_upper_bound = min(1.0, center + 0.025)
        end
    end
    
    if (y_upper_bound - y_lower_bound) < 0.1 && (y_upper_bound <= 1.0 || y_lower_bound >= 0.0)
        current_range_center = (y_lower_bound + y_upper_bound) / 2
        y_lower_bound = max(0.0, current_range_center - 0.05)
        y_upper_bound = min(1.0, current_range_center + 0.05)
    end

    y_lower_bound = 0.5
    y_upper_bound = 0.65

    y_axis_limits_segregation = (y_lower_bound, y_upper_bound)

    println("Limiti Y calcolati per la segregazione: $y_axis_limits_segregation")

    plot_title = is_aggregated ? "Evoluzione Media Indici di Segregazione" : "Evoluzione Indici di Segregazione"
    y_label = is_aggregated ? "Indice di Segregazione (Media ± SEM)" : "Indice di Segregazione"

    plt = Plots.plot( # <--- MODIFICATO: Plots.plot
        legend=:bottomright,
        #title=plot_title * title_suffix,
        xlabel="Steps",
        ylabel="Neighbor Segregation Index",
        ylims=y_axis_limits_segregation,
        #label = "Mean over $(N_SEEDS) simulations",
        
        #size=(1000, 600)
    )

    Plots.plot!(plt, data_df.step, data_df[!, col_neighbour],
        label = is_aggregated ? "Mean over 20 simulations" : "Segregazione (Neighbour)",
        linewidth = 2,
        seriesalpha = 0.8,
        color = :purple,
        line = (:solid),
        ribbon = sem_neighbour,
        fillalpha = 0.2
    )

    Plots.plot!(
    [], [], 
    seriestype = :shape,
    fillalpha = 0.2,
    color = :purple, # <-- E AGGIUNGI QUESTO
    label = "Standard Deviation"
    )

    #=Plots.plot!(plt, data_df.step, data_df[!, col_amenity_low],
        label = is_aggregated ? "Esposizione Amenità (LowIncome) Media" : "Esposizione Amenità (LowIncome)",
        linewidth = 2,
        seriesalpha = 0.8,
        color = :red,
        line = (:dash),
        ribbon = sem_amenity_low,
        fillalpha = 0.1
    )=#

    #=Plots.plot!(plt, data_df.step, data_df[!, col_amenity_high],
        label = is_aggregated ? "Esposizione Amenità (HighIncome) Media" : "Esposizione Amenità (HighIncome)",
        linewidth = 2,
        seriesalpha = 0.8,
        color = :blue,
        line = (:dot),
        ribbon = sem_amenity_high,
        fillalpha = 0.1
    )=#
    
    display(plt)
    return plt
end

function plot_zero_utility_agents_trend(data_df::DataFrame; title_suffix::String="")
    if isempty(data_df)
        println("DataFrame per il trend di agenti con utilità zero è vuoto. Impossibile generare il plot.")
        return nothing
    end

    is_aggregated = :mean_n_zero_utility in Symbol.(names(data_df))

    col_zero_utility = is_aggregated ? :mean_n_zero_utility : :zero_sat_agents
    sem_zero_utility = is_aggregated && (:sem_n_zero_utility in Symbol.(names(data_df))) ? data_df.sem_n_zero_utility : nothing

    if !(col_zero_utility in Symbol.(names(data_df)))
        println("Errore: Colonna '$col_zero_utility' necessaria per il trend di agenti con utilità zero non trovata nel DataFrame.")
        return nothing
    end

    all_data_for_ylims = Float64[]

    if !isnothing(data_df[!, col_zero_utility])
        append!(all_data_for_ylims, skipmissing(data_df[!, col_zero_utility]))
        if !isnothing(sem_zero_utility)
            append!(all_data_for_ylims, skipmissing(data_df[!, col_zero_utility] .- sem_zero_utility))
            append!(all_data_for_ylims, skipmissing(data_df[!, col_zero_utility] .+ sem_zero_utility))
        end
    end
    
    y_lower_bound = 0.0
    y_upper_bound = 0.0

    if !isempty(all_data_for_ylims) && !all(isnan, all_data_for_ylims)
        min_val = minimum(skipnan(all_data_for_ylims))
        max_val = maximum(skipnan(all_data_for_ylims))

        y_lower_bound = max(0.0, min_val - max(1.0, 0.05 * (max_val - min_val)))
        y_upper_bound = max_val + max(1.0, 0.05 * (max_val - min_val))
        
        if (y_upper_bound - y_lower_bound) < 5
            center = (y_lower_bound + y_upper_bound) / 2
            y_lower_bound = max(0.0, center - 5)
            y_upper_bound = center + 5
        end
    else
        y_upper_bound = 10.0
    end
    
    y_upper_bound = min(y_upper_bound, 1000.0)
    y_upper_bound = max(y_upper_bound, 10.0)
    
    y_axis_limits_zero_utility = (y_lower_bound, y_upper_bound)

    println("Limiti Y calcolati per agenti con utilità zero: $y_axis_limits_zero_utility")

    plot_title = is_aggregated ? "Numero Medio di Agenti con Utilità Zero" : "Numero di Agenti con Utilità Zero"
    y_label = is_aggregated ? "Numero Agenti (Media ± SEM)" : "Numero Agenti"

    plt = Plots.plot( # <--- MODIFICATO: Plots.plot
        legend=:topleft,
        title=plot_title * title_suffix,
        xlabel="Step di Simulazione",
        ylabel=y_label,
        ylims=y_axis_limits_zero_utility,
        size=(1000, 600),
        grid=true
    )

    Plots.plot!(plt, data_df.step, data_df[!, col_zero_utility],
        label = is_aggregated ? "Agenti Utilità = 0 (Media)" : "Agenti Utilità = 0",
        linewidth = 2,
        seriesalpha = 0.8,
        color = :red,
        line = (:solid),
        ribbon = sem_zero_utility,
        fillalpha = 0.1
    )
    
    display(plt)
    return plt
end

# Funzione helper per saltare NaN e Missing insieme
function skipnan(arr)
    filter(x -> !isnan(x) && !ismissing(x), arr)
end

# Nel tuo file .jl (es. Model_definitions.jl)




function plot_inner_city_median_rent_trend(data_df::DataFrame; 
                                           title_suffix::String="",
                                           y_axis_label_base::String="Median Inner City Rent (k\$)",
                                           plot_legend::Bool=true)
    
    if isempty(data_df); println("DataFrame for rent trend is empty."); return nothing; end

    is_aggregated = :mean_median_inner_city_rent in Symbol.(names(data_df))
    col_rent_line = is_aggregated ? :mean_median_inner_city_rent : :median_inner_city_rent
    col_rent_ribbon = is_aggregated && (:sem_median_inner_city_rent in Symbol.(names(data_df))) ? :sem_median_inner_city_rent : nothing

    if !(string(col_rent_line) in names(data_df)); println("Required data column is missing."); return nothing; end
    
    final_y_label = is_aggregated ? "Median Rent (Mean over Seeds)" : y_axis_label_base
    final_title = "Inner City Median Rent Evolution" * title_suffix

    valid_data = filter(row -> !ismissing(row[col_rent_line]) && !isnan(row[col_rent_line]), data_df)
    if isempty(valid_data); println("No valid data to plot."); return nothing; end
    
    ribbon_values = nothing
    if col_rent_ribbon !== nothing; ribbon_values = map(x -> (ismissing(x) || isnan(x)) ? 0.0 : x, valid_data[!, col_rent_ribbon]); end

    # --- Plot Creation ---
    plt = Plots.plot(
        legend = :topleft, # <-- Legend position
        title = final_title,
        xlabel = "Simulation Step",
        ylabel = final_y_label
    )

    Plots.plot!(plt, valid_data.step, valid_data[!, col_rent_line],
        label = is_aggregated ? "Mean Median Rent" : "Median Rent (Inner City)",
        ribbon = ribbon_values, 
        linewidth = 2, fillalpha = 0.2, marker = :circle, color = :red
    )

    # --- Adaptive Y-axis logic ---
    min_val, max_val = extrema(valid_data[!, col_rent_line])
    y_margin = (max_val - min_val) * 0.1
    Plots.ylims!(plt, (max(0, min_val - y_margin), max_val + y_margin))
    
    display(plt)
    return plt
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
    
    display(plt_income)
    return plt_income
end


function calculate_base_potential_rent(pos::NTuple{2, Int}, model::ABM)
    # Calcolo del punteggio grezzo delle amenità (logica già presente in initialize_model)
    sum_amenity = 0.0
    neighbor_positions_amenity = Agents.nearby_positions(pos, model.space, model.amenity_influence_radius)
    for np_tuple in neighbor_positions_amenity
        if model.patch_type[np_tuple...] == :public && !isnan(model.amenity_level[np_tuple...])
            distance_from_house = maximum([abs(pos[1]-np_tuple[1]), abs(pos[2]-np_tuple[2])])
            if distance_from_house > 0
                sum_amenity += model.amenity_level[np_tuple...] / distance_from_house
            end
        end
    end
    
    # Calcolo del moltiplicatore basato sulla distanza dal CBD (logica già presente in initialize_model)
    dist_from_center_for_rent = sqrt((pos[1] - model.cbd_center[1])^2 + (pos[2] - model.cbd_center[2])^2)
    dist_from_cbd_edge_for_rent = max(0.0, dist_from_center_for_rent - model.cbd_radius)
    
    max_dist_from_cbd_edge = model.max_dist_from_center_to_corner - model.cbd_radius
    if max_dist_from_cbd_edge < 1e-9; max_dist_from_cbd_edge = 1.0; end

    normalized_dist_for_rent_decay = clamp(dist_from_cbd_edge_for_rent / max_dist_from_cbd_edge, 0.0, 1.0)
    rent_distance_multiplier = 1.0 - normalized_dist_for_rent_decay * (1.0 - model.min_rent_periphery_factor)
    
    current_base_rent = model.base_potential_rent * rent_distance_multiplier
    
    # Calcolo finale del PR di base
    base_pr = current_base_rent + model.amenity_rent_factor * sum_amenity
    
    return base_pr
end


# Aggiungi queste funzioni al tuo file .jl

"""
    plot_rent_distribution(model::ABM; bins=30, title_suffix="")

Estrae tutti gli affitti capitalizzati validi dal modello e ne plotta la distribuzione.
"""
function plot_rent_histo(model::ABM; bins=30, title_suffix::String="")
    
    # Estrai i valori degli affitti validi (non-NaN) solo per le patch di tipo :house
    rent_values = Float64[]
    for i in CartesianIndices(model.patch_type)
        if model.patch_type[i] == :house && !isnan(model.capitalized_rent[i])
            push!(rent_values, model.capitalized_rent[i])
        end
    end

    if isempty(rent_values)
        println("Nessun dato di affitto valido da plottare $(title_suffix).")
        return
    end

    # Crea l'istogramma
    p_hist = stephist(rent_values,
        bins = bins,
        normalize = :pdf,
        label = "Distribuzione Affitti",
        xlabel = "Affitto Capitalizzato (\$)",
        ylabel = "Densità",
        title = "Distribuzione Affitti $(title_suffix)",
        linewidth = 2
    )
    display(p_hist)
end


"""
    plot_house_age_distribution(model::ABM; bins=30, title_suffix="")

Estrae tutte le età delle case valide dal modello e ne plotta la distribuzione.
"""
function plot_house_age_histo(model::ABM; bins=30, title_suffix::String="")

    # Estrai i valori delle età validi (non-NaN)
    age_values = Float64[]
    for i in CartesianIndices(model.patch_type)
        if model.patch_type[i] == :house && !isnan(model.house_age[i])
            push!(age_values, model.house_age[i])
        end
    end

    if isempty(age_values)
        println("Nessun dato sull'età delle case da plottare $(title_suffix).")
        return
    end

    # Crea l'istogramma
    p_hist = stephist(age_values,
        bins = bins,
        normalize = :pdf,
        label = "Distribuzione Età Case",
        xlabel = "Età Casa (step)",
        ylabel = "Densità",
        title = "Distribuzione Età Case $(title_suffix)",
        linewidth = 2
    )
    display(p_hist)
end

#=
function calculate_indexed_trends(model_df::DataFrame; base_step::Int=0)
    if isempty(model_df) || !("step" in names(model_df))
        println("DataFrame vuoto o senza colonna 'step'.")
        return model_df
    end

    # Assicurati che le colonne necessarie esistano
    required_cols = [:median_inner_city_rent, :median_income_inner_city]
    for col in required_cols
        if !(string(col) in names(model_df))
            println("ATTENZIONE: Colonna richiesta '$(col)' non trovata nel DataFrame. Impossibile calcolare l'indice.")
            return DataFrame()
        end
    end

    # Ordina per sicurezza
    df = sort(model_df, :step)
    
    # Trova i valori di base allo step specificato
    base_values_row = filter(row -> row.step == base_step, df)
    
    if isempty(base_values_row)
        println("ATTENZIONE: Nessun dato per lo step di base (step=$base_step) trovato. Impossibile calcolare l'indice.")
        return DataFrame() # Ritorna un DF vuoto
    end

    # Estrai i valori di base usando la mediana per entrambi
    base_rent = base_values_row[1, :median_inner_city_rent]
    base_income = base_values_row[1, :median_income_inner_city]

    # Filtra il DataFrame per includere solo i dati a partire dallo step di base
    df_filtered = filter(row -> row.step >= base_step, df)
    
    # Inizializza le nuove colonne
    df_filtered.indexed_rent = Vector{Union{Missing, Float64}}(missing, nrow(df_filtered))
    df_filtered.indexed_income = Vector{Union{Missing, Float64}}(missing, nrow(df_filtered))

    # Calcola l'indice per l'affitto
    if !ismissing(base_rent) && base_rent > 0
        df_filtered.indexed_rent = (df_filtered.median_inner_city_rent ./ base_rent) .* 100
    else
        println("Valore di affitto mediano di base (step $base_step) non valido. La colonna 'indexed_rent' sarà vuota.")
    end

    # Calcola l'indice per il reddito
    if !ismissing(base_income) && base_income > 0
        df_filtered.indexed_income = (df_filtered.median_income_inner_city ./ base_income) .* 100
    else
        println("Valore di reddito mediano di base (step $base_step) non valido. La colonna 'indexed_income' sarà vuota.")
    end

    return df_filtered
end =#

function calculate_indexed_trends(model_df::DataFrame; base_step::Int=0)
    if isempty(model_df) || !("step" in names(model_df))
        println("DataFrame vuoto o senza colonna 'step'.")
        return model_df
    end

    # Assicurati che le colonne necessarie esistano
    required_cols = [:median_inner_city_rent, :median_income_inner_city]
    for col in required_cols
        if !(string(col) in names(model_df))
            println("ATTENZIONE: Colonna richiesta '$(col)' non trovata nel DataFrame. Impossibile calcolare l'indice.")
            return DataFrame()
        end
    end

    # Ordina per sicurezza
    df = sort(model_df, :step)
    
    # Trova i valori di base allo step specificato
    base_values_row = filter(row -> row.step == base_step, df)
    
    if isempty(base_values_row)
        println("ATTENZIONE: Nessun dato per lo step di base (step=$base_step) trovato. Impossibile calcolare l'indice.")
        return DataFrame() # Ritorna un DF vuoto
    end

    # Estrai i valori di base usando la mediana per entrambi
    base_rent = base_values_row[1, :median_inner_city_rent]
    base_income = base_values_row[1, :median_income_inner_city]

    # Filtra il DataFrame per includere solo i dati a partire dallo step di base
    df_filtered = filter(row -> row.step >= base_step, df)
    
    # Inizializza le nuove colonne
    df_filtered.indexed_rent = Vector{Union{Missing, Float64}}(missing, nrow(df_filtered))
    df_filtered.indexed_income = Vector{Union{Missing, Float64}}(missing, nrow(df_filtered))

    # Calcola l'indice per l'affitto
    if !ismissing(base_rent) && base_rent > 0
        df_filtered.indexed_rent = (df_filtered.median_inner_city_rent ./ base_rent) .* 100
    else
        println("Valore di affitto mediano di base (step $base_step) non valido. La colonna 'indexed_rent' sarà vuota.")
    end

    # Calcola l'indice per il reddito
    if !ismissing(base_income) && base_income > 0
        df_filtered.indexed_income = (df_filtered.median_income_inner_city ./ base_income) .* 100
    else
        println("Valore di reddito mediano di base (step $base_step) non valido. La colonna 'indexed_income' sarà vuota.")
    end

    return df_filtered
end

function plot_comparison_trends(sim_df::DataFrame, real_indexed_df::DataFrame; 
                                base_step::Int, 
                                steps_per_year::Int, 
                                sim_label_base::String = "Simulazione ABM")
    
    sim_df_with_year = copy(sim_df)
    start_year_real_data = first(real_indexed_df.Year) 
    sim_df_with_year.year_sim = ((sim_df_with_year.step .- base_step) ./ steps_per_year) .+ start_year_real_data
    is_aggregated_sim = :indexed_income_sem_abs in Symbol.(names(sim_df_with_year))
    final_sim_label = is_aggregated_sim ? sim_label_base * " (Media su Seeds)" : sim_label_base

    # --- Plot Redditi (aggiornato) ---
    plt_income = Plots.plot(title="Confronto Trend Reddito Mediano Indicizzato",
                            xlabel="Anni", ylabel="Indice (Anno Base = 100)", legend=:outertopright)
    sim_income_ribbon = is_aggregated_sim ? sim_df_with_year[!, :indexed_income_sem_abs] : nothing
    Plots.plot!(plt_income, sim_df_with_year.year_sim, sim_df_with_year.indexed_income, label=final_sim_label, ribbon=sim_income_ribbon, linewidth=3, color=:red, marker=:circle, fillalpha=0.2)
    #Plots.plot!(plt_income, real_indexed_df.Year, real_indexed_df.indexed_la_income, label="Los Angeles (90014)", linestyle=:dot, color=:gray) # Commentato come richiesto
    Plots.plot!(plt_income, real_indexed_df.Year, real_indexed_df.indexed_brooklyn_income, label="Brooklyn (11216)", linestyle=:dash)
    Plots.plot!(plt_income, real_indexed_df.Year, real_indexed_df.indexed_dc_income, label="Washington D.C. (20001)", linestyle=:dot)
    Plots.plot!(plt_income, real_indexed_df.Year, real_indexed_df.indexed_houston_income, label="Houston (77003)", linestyle=:dashdot)
    Plots.plot!(plt_income, real_indexed_df.Year, real_indexed_df.indexed_austin_income, label="Austin (78702)", linestyle=:solid)
    Plots.plot!(plt_income, real_indexed_df.Year, real_indexed_df.indexed_philly_income, label="Philadelphia (19123)", linestyle=:dash)

    # --- Plot Affitti (aggiornato) ---
    plt_rent = Plots.plot(title="Confronto Trend Affitto Mediano Indicizzato",
                          xlabel="Anni", ylabel="Indice (Anno Base = 100)", legend=:outertopright)
    sim_rent_ribbon = is_aggregated_sim ? sim_df_with_year[!, :indexed_rent_sem_abs] : nothing
    Plots.plot!(plt_rent, sim_df_with_year.year_sim, sim_df_with_year.indexed_rent, label=final_sim_label, ribbon=sim_rent_ribbon, linewidth=3, color=:blue, marker=:circle, fillalpha=0.2)
    #Plots.plot!(plt_rent, real_indexed_df.Year, real_indexed_df.indexed_la_rent, label="Los Angeles (90014)", linestyle=:dot, color=:gray) # Commentato come richiesto
    Plots.plot!(plt_rent, real_indexed_df.Year, real_indexed_df.indexed_brooklyn_rent, label="Brooklyn (11216)", linestyle=:dash)
    Plots.plot!(plt_rent, real_indexed_df.Year, real_indexed_df.indexed_dc_rent, label="Washington D.C. (20001)", linestyle=:dot)
    Plots.plot!(plt_rent, real_indexed_df.Year, real_indexed_df.indexed_houston_rent, label="Houston (77003)", linestyle=:dashdot)
    Plots.plot!(plt_rent, real_indexed_df.Year, real_indexed_df.indexed_austin_rent, label="Austin (78702)", linestyle=:solid)
    Plots.plot!(plt_rent, real_indexed_df.Year, real_indexed_df.indexed_philly_rent, label="Philadelphia (19123)", linestyle=:dash)
    
    final_plot = Plots.plot(plt_income, plt_rent, layout=(1, 2), size=(1800, 700))
    display(final_plot)
    return final_plot
end

#=function plot_single_city_comparison(model_before::ABM, model_after::ABM, plot_type::Symbol)
    
    title_fragment = ""
    primary_value_field_symbol = :house_age 
    plot_range_field_primary_symbol = :plot_age_range

    if plot_type == :age
        title_fragment = "Età Case"
        primary_value_field_symbol = :house_age
        plot_range_field_primary_symbol = :plot_age_range
    elseif plot_type == :cap_rent
        title_fragment = "Affitto Capitalizzato"
        primary_value_field_symbol = :capitalized_rent
        plot_range_field_primary_symbol = :plot_rent_range
    elseif plot_type == :pot_rent
        title_fragment = "Affitto Potenziale"
        primary_value_field_symbol = :potential_rent
        plot_range_field_primary_symbol = :plot_potential_rent_range
    else
        error("Tipo di plot non riconosciuto: $plot_type. Usare :age, :cap_rent, o :pot_rent.")
    end

    println("Generazione plot di confronto per: $title_fragment (prima vs. dopo)...")
    fig = Figure(size = (1300, 700)) 

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

    ac_before = _get_agent_color_func(model_before)
    ac_after = _get_agent_color_func(model_after)

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

        final_combined_colormap = [RGBA(1.0,1.0,1.0,1.0); primary_sub_colormap; amenity_sub_colormap; RGBA(0.2,0.2,0.2,1.0)]
        id_empty=1; offset_after_primary=id_empty+N_primary_bins; offset_after_amenity=offset_after_primary+N_amenity_bins; id_cbd=offset_after_amenity+1;
        total_entries = length(final_combined_colormap)

        heatarray_func = function(model_from_abmplot::ABM)
            patch_type_matrix = model_from_abmplot.patch_type 
            patch_ids = zeros(Int, size(patch_type_matrix)) 
            primary_value_matrix = getproperty(model_from_abmplot, p_value_field_symbol) 
            amenity_level_matrix = model_from_abmplot.amenity_level
            for idx in CartesianIndices(patch_type_matrix)
                patch_sym = patch_type_matrix[idx] 
                if patch_sym == :house
                    primary_val = primary_value_matrix[idx]
                    range_primary = max_primary_plot - min_primary_plot 
                    norm_primary = (!isnan(primary_val) && range_primary > 1e-9) ? clamp((primary_val - min_primary_plot) / range_primary, 0.0, 1.0) : 0.0
                    bin_idx_primary = clamp(floor(Int, norm_primary * (N_primary_bins - 1e-9)), 0, N_primary_bins - 1)
                    patch_ids[idx] = id_empty + 1 + bin_idx_primary
                elseif patch_sym == :public
                    amenity_val = amenity_level_matrix[idx]
                    range_amenity = max_amenity_plot - min_amenity_plot
                    norm_amenity = (!isnan(amenity_val) && range_amenity > 1e-9) ? clamp((amenity_val - min_amenity_plot) / range_amenity, 0.0, 1.0) : 0.0
                    bin_idx_amenity = clamp(floor(Int, norm_amenity * (N_amenity_bins - 1e-9)), 0, N_amenity_bins - 1)
                    patch_ids[idx] = offset_after_primary + 1 + bin_idx_amenity
                elseif patch_sym == :empty; patch_ids[idx] = id_empty;
                elseif patch_sym == :cbd; patch_ids[idx] = id_cbd;
                else; patch_ids[idx] = id_empty; end
            end
            return patch_ids
        end
        
        return (heatarray=heatarray_func, 
                heatkwargs=(colormap=final_combined_colormap, colorrange=(1, total_entries)),
                primary_cmap_details=(cmap=primary_sub_colormap, lims=(min_primary_plot, max_primary_plot)),
                amenity_cmap_details=(cmap=amenity_sub_colormap, lims=(min_amenity_plot, max_amenity_plot))
               )
    end

    # --- Plot "Prima" ---
    ax_before = Axis(fig[1, 1], title="$title_fragment (Prima Dinamica)", aspect=DataAspect())
    vis_data_before = _prepare_heatmap_visuals(model_before, plot_type, primary_value_field_symbol, plot_range_field_primary_symbol)
    abmplot!(ax_before, model_before; 
             agent_color=ac_before, agent_size=10,
             heatarray=vis_data_before.heatarray, heatkwargs=vis_data_before.heatkwargs,
             add_colorbar=false)
    hidedecorations!(ax_before)
    # --- AGGIUNTA: Disegna rettangolo inner city per "Prima" ---
    try
        r_start, r_end, c_start, c_end = get_inner_city_bounds(model_before) # Usa model_before
        x_coords = [c_start - 0.5, c_end + 0.5, c_end + 0.5, c_start - 0.5, c_start - 0.5]
        y_coords = [r_start - 0.5, r_start - 0.5, r_end + 0.5, r_end + 0.5, r_start - 0.5]
        Makie.lines!(ax_before, x_coords, y_coords, color = :black, linewidth = 2, linestyle = :dash)
    catch e_rect
        println("ERRORE disegno rettangolo Inner City (Prima): "); showerror(stdout, e_rect); println()
    end
    # --- FINE AGGIUNTA ---


    # --- Plot "Dopo" ---
    ax_after = Axis(fig[1, 2], title="$title_fragment (Dopo Dinamica)", aspect=DataAspect())
    vis_data_after = _prepare_heatmap_visuals(model_after, plot_type, primary_value_field_symbol, plot_range_field_primary_symbol)
    abmplot!(ax_after, model_after;
             agent_color=ac_after, agent_size=10,
             heatarray=vis_data_after.heatarray, heatkwargs=vis_data_after.heatkwargs,
             add_colorbar=false)
    hidedecorations!(ax_after)
    # --- AGGIUNTA: Disegna rettangolo inner city per "Dopo" ---
    try
        r_start, r_end, c_start, c_end = get_inner_city_bounds(model_after) # Usa model_after
        x_coords = [c_start - 0.5, c_end + 0.5, c_end + 0.5, c_start - 0.5, c_start - 0.5]
        y_coords = [r_start - 0.5, r_start - 0.5, r_end + 0.5, r_end + 0.5, r_start - 0.5]
        Makie.lines!(ax_after, x_coords, y_coords, color = :black, linewidth = 2, linestyle = :dash)
    catch e_rect
        println("ERRORE disegno rettangolo Inner City (Dopo): "); showerror(stdout, e_rect); println()
    end
    # --- FINE AGGIUNTA ---


    # --- Colorbars (come prima) ---
    all_incomes_after = [a.income_value for a in allagents(model_after) if hasproperty(a, :income_value) && isa(a.income_value, Real)]
    min_inc_plot = isempty(all_incomes_after) ? 0.0 : minimum(all_incomes_after)
    max_inc_plot = isempty(all_incomes_after) ? 1.0 : (min_inc_plot == maximum(all_incomes_after) ? min_inc_plot + 1.0 : maximum(all_incomes_after))
    
    num_cb_samples = 50
    low_income_cb_colors = [get_low_income_color_from_norm(n) for n in range(0, 1, length=num_cb_samples)]
    high_income_cb_colors = [get_high_income_color_from_norm(n) for n in range(0, 1, length=num_cb_samples)]

    Colorbar(fig[1, 3], colormap=low_income_cb_colors, limits=(min_inc_plot, max_inc_plot), label="Reddito LowIncome", width=20, height=Relative(0.23)) 
    Colorbar(fig[1, 3], colormap=high_income_cb_colors, limits=(min_inc_plot, max_inc_plot), label="Reddito HighIncome", width=20, height=Relative(0.23), tellheight=false, halign=:right, valign = :bottom)
    Colorbar(fig[1, 3], colormap=vis_data_after.primary_cmap_details.cmap, limits=vis_data_after.primary_cmap_details.lims, label=title_fragment, width=20, height=Relative(0.23), tellheight=false, valign=:center, halign=:left )
    Colorbar(fig[1, 3], colormap=vis_data_after.amenity_cmap_details.cmap, limits=vis_data_after.amenity_cmap_details.lims, label="Livello Amenità", width=20, height=Relative(0.23), tellheight=false, valign=:top, halign=:center)

    colgap!(fig.layout, 1, 5) 
    colgap!(fig.layout, 2, 15)
    colsize!(fig.layout, 1, Relative(0.42))
    colsize!(fig.layout, 2, Relative(0.42))
    colsize!(fig.layout, 3, Relative(0.16))
    
    display(fig)
    return fig
end=#

#=
function plot_single_city_comparison(model_before::ABM, model_after::ABM, plot_type::Symbol, title_fontsize::Int = 28)
    
    title_fragment = ""
    primary_value_field_symbol = :house_age 
    plot_range_field_primary_symbol = :plot_age_range

    if plot_type == :age
        title_fragment = "House Age"
        primary_value_field_symbol = :house_age
        plot_range_field_primary_symbol = :plot_age_range
    elseif plot_type == :cap_rent
        title_fragment = "Capitilized Rent"
        primary_value_field_symbol = :capitalized_rent
        plot_range_field_primary_symbol = :plot_rent_range
    elseif plot_type == :pot_rent
        title_fragment = "Potential Rent"
        primary_value_field_symbol = :potential_rent
        plot_range_field_primary_symbol = :plot_potential_rent_range
    else
        error("Tipo di plot non riconosciuto: $plot_type. Usare :age, :cap_rent, o :pot_rent.")
    end

    println("Generazione plot di confronto per: $title_fragment (prima vs. dopo)...")
    fig = Figure(size = (900, 1400)) 

  

    ac_before = _get_agent_color_func(model_before)
    ac_after = _get_agent_color_func(model_after)

    

    # --- Plot "Prima" ---
    ax_before = Axis(fig[1, 1], title="$title_fragment (Before Developer Intervention)", aspect=DataAspect())
    vis_data_before = _prepare_heatmap_visuals(model_before, plot_type, primary_value_field_symbol, plot_range_field_primary_symbol)
    abmplot!(ax_before, model_before; 
             agent_color=ac_before, agent_size=8,
             heatarray=vis_data_before.heatarray, heatkwargs=vis_data_before.heatkwargs,
             add_colorbar=false)
    hidedecorations!(ax_before)
    try
        r_start, r_end, c_start, c_end = get_inner_city_bounds(model_before)
        Makie.lines!(ax_before, [c_start - 0.5, c_end + 0.5, c_end + 0.5, c_start - 0.5, c_start - 0.5], 
                                [r_start - 0.5, r_start - 0.5, r_end + 0.5, r_end + 0.5, r_start - 0.5], 
                                color = :black, linewidth = 2, linestyle = :dash)
    catch e_rect
        println("ERRORE disegno rettangolo Inner City (Prima): "); showerror(stdout, e_rect); println()
    end


    # --- Plot "Dopo" ---
    ax_after = Axis(fig[1, 2], title="$title_fragment (After Some steps)", aspect=DataAspect())
    vis_data_after = _prepare_heatmap_visuals(model_after, plot_type, primary_value_field_symbol, plot_range_field_primary_symbol)
    abmplot!(ax_after, model_after;
             agent_color=ac_after, agent_size=8,
             heatarray=vis_data_after.heatarray, heatkwargs=vis_data_after.heatkwargs,
             add_colorbar=false)
    hidedecorations!(ax_after)
    try
        r_start, r_end, c_start, c_end = get_inner_city_bounds(model_after)
        Makie.lines!(ax_after, [c_start - 0.5, c_end + 0.5, c_end + 0.5, c_start - 0.5, c_start - 0.5],
                                [r_start - 0.5, r_start - 0.5, r_end + 0.5, r_end + 0.5, r_start - 0.5],
                                color = :black, linewidth = 2, linestyle = :dash)
    catch e_rect
        println("ERRORE disegno rettangolo Inner City (Dopo): "); showerror(stdout, e_rect); println()
    end

    legend_grid = GridLayout(fig[1:2, 2])

#=
    # --- Colorbars e Legende ---
    all_incomes_after = [a.income_value for a in allagents(model_after) if hasproperty(a, :income_value) && isa(a.income_value, Real)]
    min_inc_plot = isempty(all_incomes_after) ? 0.0 : minimum(all_incomes_after)
    max_inc_plot = isempty(all_incomes_after) ? 1.0 : (min_inc_plot == maximum(all_incomes_after) ? min_inc_plot + 1.0 : maximum(all_incomes_after))
    
    num_cb_samples = 50
    low_income_cb_colors = [get_low_income_color_from_norm(n) for n in range(0, 1, length=num_cb_samples)]
    high_income_cb_colors = [get_high_income_color_from_norm(n) for n in range(0, 1, length=num_cb_samples)]

    # Colobar per Income Low
    Colorbar(fig[1, 3], colormap=low_income_cb_colors, limits=(min_inc_plot, max_inc_plot), label="Reddito LowIncome", width=20, height=Relative(0.23)) 
    # Colobar per Income High
    Colorbar(fig[1, 3], colormap=high_income_cb_colors, limits=(min_inc_plot, max_inc_plot), label="Reddito HighIncome", width=20, height=Relative(0.23), tellheight=false, halign=:right, valign = :bottom)
    # Colobar per Primary value (Age/Rent)
    Colorbar(fig[1, 3], colormap=vis_data_after.primary_cmap_details.cmap, limits=vis_data_after.primary_cmap_details.lims, label=title_fragment, width=20, height=Relative(0.23), tellheight=false, valign=:center, halign=:left )
    # Colobar per Amenity
    Colorbar(fig[1, 3], colormap=vis_data_after.amenity_cmap_details.cmap, limits=vis_data_after.amenity_cmap_details.lims, label="Livello Amenità", width=20, height=Relative(0.23), tellheight=false, valign=:top, halign=:center)

    # --- NUOVA LEGENDA per Public Housing ---
    # Posiziona l'elemento legenda in basso a sinistra (o dove è appropriato)
    # Crea una lista di elementi legenda per Makie.Legend
    legend_elements = [
        MarkerElement(marker = '◼', color = vis_data_after.public_housing_color, marker_size = 20),
    ]
    Legend(fig[1, 3], legend_elements, ["Public Housing"], tellheight=false, tellwidth=false, halign = :right, valign = :center, orientation = :vertical, patchsize = (35, 10))


    colgap!(fig.layout, 1, 5) 
    colgap!(fig.layout, 2, 15)
    colsize!(fig.layout, 1, Relative(0.42))
    colsize!(fig.layout, 2, Relative(0.42))
    colsize!(fig.layout, 3, Relative(0.16))=#
    
    # --- 4. Aggiusta il Layout Finale ---
    colsize!(fig.layout, 2, Relative(0.25)) # Assegna il 25% della larghezza alla colonna delle legende
    rowgap!(fig.layout, 1, 15) # Spazio tra i due plot
    
    display(fig)
    return fig
end
=#


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

    display(fig)
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


function plot_single_city_comparison(model_before::ABM, model_after::ABM, plot_type::Symbol)
    
    title_fragment = ""
    primary_value_field_symbol = :house_age 
    plot_range_field_primary_symbol = :plot_age_range

    if plot_type == :age
        title_fragment = "House Age"
        primary_value_field_symbol = :house_age
        plot_range_field_primary_symbol = :plot_age_range
    elseif plot_type == :cap_rent
        title_fragment = "Capitilized Rent"
        primary_value_field_symbol = :capitalized_rent
        plot_range_field_primary_symbol = :plot_rent_range
    elseif plot_type == :pot_rent
        title_fragment = "Potential Rent"
        primary_value_field_symbol = :potential_rent
        plot_range_field_primary_symbol = :plot_potential_rent_range
    else
        error("Tipo di plot non riconosciuto: $plot_type. Usare :age, :cap_rent, o :pot_rent.")
    end

    println("Generazione plot di confronto per: $title_fragment (prima vs. dopo)...")
    fig = Figure(size = (700, 1300)) 

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

    ac_before = _get_agent_color_func(model_before)
    ac_after = _get_agent_color_func(model_after)

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

    # --- Plot "Prima" ---
    ax_before = Axis(fig[1, 1], title="$title_fragment (Before Developer Intervention)",#=titlesize= 12,=# aspect=DataAspect())
    vis_data_before = _prepare_heatmap_visuals(model_before, plot_type, primary_value_field_symbol, plot_range_field_primary_symbol)
    abmplot!(ax_before, model_before; 
             agent_color=ac_before, agent_size=10,
             heatarray=vis_data_before.heatarray, heatkwargs=vis_data_before.heatkwargs,
             add_colorbar=false)
    hidedecorations!(ax_before)
    try
        r_start, r_end, c_start, c_end = get_inner_city_bounds(model_before)
        Makie.lines!(ax_before, [c_start - 0.5, c_end + 0.5, c_end + 0.5, c_start - 0.5, c_start - 0.5], 
                                [r_start - 0.5, r_start - 0.5, r_end + 0.5, r_end + 0.5, r_start - 0.5], 
                                color = :black, linewidth = 2, linestyle = :dash)
    catch e_rect
        println("ERRORE disegno rettangolo Inner City (Prima): "); showerror(stdout, e_rect); println()
    end


    # --- Plot "Dopo" ---
    ax_after = Axis(fig[1, 2], title="$title_fragment (After Some steps)",#=titlesize= 12,=# aspect=DataAspect())
    vis_data_after = _prepare_heatmap_visuals(model_after, plot_type, primary_value_field_symbol, plot_range_field_primary_symbol)
    abmplot!(ax_after, model_after;
             agent_color=ac_after, agent_size=10,
             heatarray=vis_data_after.heatarray, heatkwargs=vis_data_after.heatkwargs,
             add_colorbar=false)
    hidedecorations!(ax_after)
    try
        r_start, r_end, c_start, c_end = get_inner_city_bounds(model_after)
        Makie.lines!(ax_after, [c_start - 0.5, c_end + 0.5, c_end + 0.5, c_start - 0.5, c_start - 0.5],
                                [r_start - 0.5, r_start - 0.5, r_end + 0.5, r_end + 0.5, r_start - 0.5],
                                color = :black, linewidth = 2, linestyle = :dash)
    catch e_rect
        println("ERRORE disegno rettangolo Inner City (Dopo): "); showerror(stdout, e_rect); println()
    end


    # --- Colorbars e Legende ---
    all_incomes_after = [a.income_value for a in allagents(model_after) if hasproperty(a, :income_value) && isa(a.income_value, Real)]
    min_inc_plot = isempty(all_incomes_after) ? 0.0 : minimum(all_incomes_after)
    max_inc_plot = isempty(all_incomes_after) ? 1.0 : (min_inc_plot == maximum(all_incomes_after) ? min_inc_plot + 1.0 : maximum(all_incomes_after))
    
    num_cb_samples = 50
    low_income_cb_colors = [get_low_income_color_from_norm(n) for n in range(0, 1, length=num_cb_samples)]
    high_income_cb_colors = [get_high_income_color_from_norm(n) for n in range(0, 1, length=num_cb_samples)]

    # Colobar per Income Low
    Colorbar(fig[1, 3], colormap=low_income_cb_colors, limits=(min_inc_plot, max_inc_plot), label="Reddito LowIncome", width=20, height=Relative(0.23)) 
    # Colobar per Income High
    Colorbar(fig[1, 3], colormap=high_income_cb_colors, limits=(min_inc_plot, max_inc_plot), label="Reddito HighIncome", width=20, height=Relative(0.23), tellheight=false, halign=:right, valign = :bottom)
    # Colobar per Primary value (Age/Rent)
    Colorbar(fig[1, 3], colormap=vis_data_after.primary_cmap_details.cmap, limits=vis_data_after.primary_cmap_details.lims, label=title_fragment, width=20, height=Relative(0.23), tellheight=false, valign=:center, halign=:left )
    # Colobar per Amenity
    Colorbar(fig[1, 3], colormap=vis_data_after.amenity_cmap_details.cmap, limits=vis_data_after.amenity_cmap_details.lims, label="Livello Amenità", width=20, height=Relative(0.23), tellheight=false, valign=:top, halign=:center)

    # --- NUOVA LEGENDA per Public Housing ---
    # Posiziona l'elemento legenda in basso a sinistra (o dove è appropriato)
    # Crea una lista di elementi legenda per Makie.Legend
    legend_elements = [
        MarkerElement(marker = '◼', color = vis_data_after.public_housing_color, marker_size = 20),
    ]
    Legend(fig[1, 3], legend_elements, ["Public Housing"], tellheight=false, tellwidth=false, halign = :right, valign = :center, orientation = :vertical, patchsize = (35, 10))


    colgap!(fig.layout, 1, 5) 
    colgap!(fig.layout, 2, 15)
    colsize!(fig.layout, 1, Relative(0.42))
    colsize!(fig.layout, 2, Relative(0.42))
    colsize!(fig.layout, 3, Relative(0.16))
    
    display(fig)
    return fig
end

function calculate_indexed_trends_aggregate(summary_df::DataFrame;
                                            base_step::Int,
                                            step_col::Symbol = :step,
                                            mean_income_col::Symbol, # Es. :mean_median_income_inner_city
                                            sem_income_col::Symbol,  # Es. :sem_median_income_inner_city
                                            mean_rent_col::Symbol,   # Es. :mean_median_inner_city_rent
                                            sem_rent_col::Symbol)    # Es. :sem_median_inner_city_rent
    
    # Seleziona e rinomina le colonne per generalizzare la logica interna
    # Assicurati che tutte le colonne necessarie esistano in summary_df
    cols_to_select = [step_col, mean_income_col, sem_income_col, mean_rent_col, sem_rent_col]
    missing_cols = filter(c -> !(string(c) in names(summary_df)), cols_to_select)
    if !isempty(missing_cols)
        error("Colonne mancanti in summary_df: $missing_cols")
        return DataFrame()
    end

    df_processed = select(summary_df, cols_to_select...)
    # Rinomina per standardizzare internamente e facilitare i calcoli
    rename!(df_processed, 
            mean_income_col => :current_mean_income, sem_income_col => :current_sem_income,
            mean_rent_col   => :current_mean_rent,   sem_rent_col   => :current_sem_rent)

    base_row_idx = findfirst(==(base_step), df_processed[!, step_col])
    if isnothing(base_row_idx)
        @warn "Base step $base_step non trovato nel DataFrame aggregato. Impossibile indicizzare."
        # Restituisce un DataFrame vuoto con la struttura attesa per evitare errori downstream
        return DataFrame(
            step_col => Int[], 
            :indexed_income => Float64[], :indexed_income_sem_abs => Float64[],
            :indexed_rent => Float64[], :indexed_rent_sem_abs => Float64[]
        )
    end

    base_mean_income = df_processed[base_row_idx, :current_mean_income]
    base_mean_rent = df_processed[base_row_idx, :current_mean_rent]

    # Controlla valori base problematici
    if ismissing(base_mean_income) || base_mean_income == 0 || isnan(base_mean_income) || 
       ismissing(base_mean_rent) || base_mean_rent == 0 || isnan(base_mean_rent)
        @warn """Valore base per income ($base_mean_income) o rent ($base_mean_rent) 
                 allo step $base_step è zero, NaN o missing. L'indicizzazione potrebbe non essere affidabile."""
        # Comunque, procedi con il calcolo; i NaN si propagheranno
    end
    
    # Calcola i valori indicizzati per le medie
    df_processed[!, :indexed_income] = (df_processed.current_mean_income ./ base_mean_income) .* 100
    df_processed[!, :indexed_rent] = (df_processed.current_mean_rent ./ base_mean_rent) .* 100

    # Scala i SEM in modo proporzionale (SEM indicizzato)
    # Usa abs() al denominatore per evitare problemi con valori base negativi (anche se improbabile per income/rent)
    df_processed[!, :indexed_income_sem_abs] = (df_processed.current_sem_income ./ abs(base_mean_income)) .* 100
    df_processed[!, :indexed_rent_sem_abs] = (df_processed.current_sem_rent ./ abs(base_mean_rent)) .* 100
    
    # --- MODIFICA CHIAVE: Filtra il DataFrame per includere solo gli step >= base_step ---
    df_filtered_after_indexing = filter(row -> row[step_col] >= base_step, df_processed)
    
    # Seleziona e restituisce solo le colonne rilevanti dal DataFrame filtrato
    # È importante rinominare step_col di nuovo al suo nome originale se diverso da :step
    # per coerenza con l'output atteso da plot_comparison_trends.
    # Tuttavia, plot_comparison_trends si aspetta una colonna chiamata ':step'.
    # Se step_col è diverso, assicurati che il DataFrame finale abbia una colonna ':step'.
    # Qui assumiamo che step_col sia ':step' o che il rename sia gestito a monte/valle se necessario.
    # Per ora, ci assicuriamo che la colonna step_col sia presente nell'output.
    
    final_df = select(df_filtered_after_indexing, 
                      step_col, # Mantiene il nome originale della colonna step
                      :indexed_income, :indexed_income_sem_abs, 
                      :indexed_rent, :indexed_rent_sem_abs)

    # Se la colonna step originale non era chiamata ':step', rinominala a ':step'
    # perché plot_comparison_trends si aspetta `sim_df.step`
    if step_col != :step
        rename!(final_df, step_col => :step)
    end
                      
    return final_df
end



function run_and_plot_scenario(seeds_to_run::Vector{Int},
                               real_data_df::DataFrame,
                               plot_title_suffix::String;
                               # Parametri di simulazione con valori di default
                               grid_dims = (40, 40),
                               num_agents = 800,
                               n_steps_termalizzazione = 15,
                               n_steps_dinamici = 112,
                               steps_per_year = 8)

    println("\n\n#########################################################################")
    println("--- INIZIO SCENARIO: $plot_title_suffix ---")
    println("--- Seed da simulare: $seeds_to_run ---")
    println("#########################################################################\n")

    # --- 1. PREPARAZIONE E CONFIGURAZIONE (come nel tuo script) ---
    current_theoretical_income_dist = create_mixed_distribution()
    abm_w_poor_param = probs(current_theoretical_income_dist.prior)[1]
    all_indexed_sim_results = Dict{Int, DataFrame}()

    if !@isdefined(real_data_df)
        println("ATTENZIONE: 'real_data_df' non trovato. Assicurati che sia disponibile.")
        return
    end

    # --- 2. CICLO DI ESECUZIONE PER OGNI SEED ---
    for current_seed in seeds_to_run
        println("\n--- Inizio Esecuzione per Seed: $current_seed ---")

        params_for_current_run = (
            dims = grid_dims,
            N_agents = num_agents,
            income_distribution = current_theoretical_income_dist,
            w_poor = abm_w_poor_param,
            seed = current_seed
        )

        local initial_model = initialize_model(; params_for_current_run...)
        _, model_metrics_df_term, thermalized_model_object = run_simulation_phase_and_collect_data(initial_model, n_steps_termalizzazione, agent_step!, model_step_termalizzazione!, 0, "Termalizzazione")
        _, model_metrics_df_din, _ = run_simulation_phase_and_collect_data(thermalized_model_object, n_steps_dinamici, agent_step!, model_step_dinamica!, n_steps_termalizzazione, "Dinamica Ambientale")

        local combined_model_metrics_df = vcat(model_metrics_df_term, model_metrics_df_din[2:end,:])
        local start_of_dynamic_phase_step = n_steps_termalizzazione

        local indexed_sim_data_df = calculate_indexed_trends(
            combined_model_metrics_df,
            base_step=start_of_dynamic_phase_step
        )

        if !isnothing(indexed_sim_data_df) && !isempty(indexed_sim_data_df)
            all_indexed_sim_results[current_seed] = indexed_sim_data_df
            println("Risultati indicizzati per il seed $current_seed salvati correttamente.")
        else
            println("ATTENZIONE: DataFrame indicizzato vuoto per il seed $current_seed. Salto.")
        end
    end

    println("\n--- TUTTE LE SIMULAZIONI PER LO SCENARIO '$plot_title_suffix' COMPLETATE ---")

    # --- 3. BLOCCO DI PLOTTING FINALE (MULTI-SEED vs DATI REALI) ---
    if !isempty(all_indexed_sim_results)
        println("\n--- Generazione del grafico di confronto: $plot_title_suffix ---")

        local base_step = n_steps_termalizzazione
        local start_year_real_data = first(real_data_df.Year)

        # PLOT REDDITO
        plt_income = Plots.plot(title="Trend Reddito Mediano " * plot_title_suffix, xlabel="Anni", ylabel="Indice (Anno Base = 100)", legend=:outertopright, size=(1800, 700))
        Plots.plot!(plt_income, real_data_df.Year, real_data_df.indexed_brooklyn_income, label="Brooklyn (11216)", linestyle=:dash, linewidth=2)
        Plots.plot!(plt_income, real_data_df.Year, real_data_df.indexed_dc_income, label="Washington D.C. (20001)", linestyle=:dot, linewidth=2)
        Plots.plot!(plt_income, real_data_df.Year, real_data_df.indexed_houston_income, label="Houston (77003)", linestyle=:dashdot, linewidth=2)
        Plots.plot!(plt_income, real_data_df.Year, real_data_df.indexed_austin_income, label="Austin (78702)", linestyle=:solid, linewidth=2)
        Plots.plot!(plt_income, real_data_df.Year, real_data_df.indexed_philly_income, label="Philadelphia (19123)", linestyle=:dash, linewidth=2)

        # PLOT AFFITTO
        plt_rent = Plots.plot(title="Trend Affitto Mediano " * plot_title_suffix, xlabel="Anni", ylabel="Indice (Anno Base = 100)", legend=:outertopright)
        Plots.plot!(plt_rent, real_data_df.Year, real_data_df.indexed_brooklyn_rent, label="Brooklyn (11216)", linestyle=:dash, linewidth=2)
        Plots.plot!(plt_rent, real_data_df.Year, real_data_df.indexed_dc_rent, label="Washington D.C. (20001)", linestyle=:dot, linewidth=2)
        Plots.plot!(plt_rent, real_data_df.Year, real_data_df.indexed_houston_rent, label="Houston (77003)", linestyle=:dashdot, linewidth=2)
        Plots.plot!(plt_rent, real_data_df.Year, real_data_df.indexed_austin_rent, label="Austin (78702)", linestyle=:solid, linewidth=2)
        Plots.plot!(plt_rent, real_data_df.Year, real_data_df.indexed_philly_rent, label="Philadelphia (19123)", linestyle=:dash, linewidth=2)

        sim_colors = [:red, :orange, :purple, :cyan, :magenta, :brown, :black, :green, :gold]
        sim_styles = [:solid, :dash, :dot, :dashdot]

        for (i, (seed, sim_df)) in enumerate(all_indexed_sim_results)
            local sim_df_with_year = copy(sim_df)
            sim_df_with_year.year_sim = ((sim_df_with_year.step .- base_step) ./ steps_per_year) .+ start_year_real_data
            
            local color_idx = (i - 1) % length(sim_colors) + 1
            local style_idx = (i - 1) % length(sim_styles) + 1
            
            println("Aggiungo al plot la simulazione con seed: $seed")
            
            Plots.plot!(plt_income, sim_df_with_year.year_sim, sim_df_with_year.indexed_income, label="Sim (Seed $seed)", linewidth=3.5, color=sim_colors[color_idx], linestyle=sim_styles[style_idx], marker=:circle, markersize=3)
            Plots.plot!(plt_rent, sim_df_with_year.year_sim, sim_df_with_year.indexed_rent, label="Sim (Seed $seed)", linewidth=3.5, color=sim_colors[color_idx], linestyle=sim_styles[style_idx], marker=:circle, markersize=3)
        end
        
        final_comparison_plot = Plots.plot(plt_income, plt_rent, layout=(1, 2))
        display(final_comparison_plot)
    else
        println("Nessun risultato di simulazione indicizzato da plottare per lo scenario: ", plot_title_suffix)
    end
    println("\n--- FINE SCENARIO: $plot_title_suffix ---\n")
end

function calculate_moving_average(data::Vector{<:Real}, window_size::Int)
    if window_size > length(data) || window_size < 1
        # Restituisce un array vuoto se la finestra è troppo grande o non valida
        return Float64[] 
    end
    # Usa una list comprehension per calcolare le medie
    return [mean(data[i:(i + window_size - 1)]) for i in 1:(length(data) - window_size + 1)]
end


#=function plot_calibration_trends(
    simulation_df::DataFrame,
    real_data_df::DataFrame;
    sim_rent_col::Symbol = :median_rent_inner_city,
    sim_income_col::Symbol = :median_income_inner_city,
    steps_per_year::Int = 8,
    moving_avg_window::Int = 5,
    simulation_seed::Int = 0
)
    println("--- Avvio processo di calibrazione visuale ---")
    println("Ipotesi: $(steps_per_year) steps/anno, Finestra Media Mobile: $(moving_avg_window) anni")

    # --- 1. Aggregazione dati simulazione in "Anni" ---
    # Escludiamo la fase di termalizzazione se presente (step < 0 o step iniziale)
    sim_data_dynamic = filter(row -> row.step > 0, simulation_df)
    
    # Calcola a quale "anno" appartiene ogni step
    sim_data_dynamic.year = floor.(Int, (sim_data_dynamic.step .- 1) ./ steps_per_year) .+ 1
    
    # Raggruppa per anno e calcola la media delle metriche per quell'anno
    yearly_sim_data = combine(groupby(sim_data_dynamic, :year),
        sim_rent_col => mean,
        sim_income_col => mean,
        renamecols=false
    )

    # Ordina per sicurezza
    sort!(yearly_sim_data, :year)
    println("Dati simulati aggregati in $(nrow(yearly_sim_data)) 'anni'.")

    # --- 2. Calcolo Medie Mobili e Indicizzazione ---
    # Applica la media mobile ai dati annuali simulati
    sim_rent_ma = calculate_moving_average(yearly_sim_data[!, sim_rent_col], moving_avg_window)
    sim_income_ma = calculate_moving_average(yearly_sim_data[!, sim_income_col], moving_avg_window)

    if isempty(sim_rent_ma) || isempty(sim_income_ma)
        println("ERRORE: Dati simulati insufficienti per calcolare una media mobile con finestra $(moving_avg_window).")
        println("Anni simulati necessari: $(moving_avg_window), anni disponibili: $(nrow(yearly_sim_data)).")
        return
    end

    # Indicizza i dati della media mobile (Base 100 sul primo valore)
    indexed_sim_rent = (sim_rent_ma ./ sim_rent_ma[1]) .* 100
    indexed_sim_income = (sim_income_ma ./ sim_income_ma[1]) .* 100

    # Prepara l'asse X per i dati simulati
    # La media mobile parte dall'anno `moving_avg_window`-esimo
    start_year_real = minimum(real_data_df.Year)
    sim_years_axis = (start_year_real + moving_avg_window - 1):(start_year_real + moving_avg_window - 2 + length(indexed_sim_rent))

    # --- 3. Plotting ---
    
    # Plot Affitti
    p_rent = Plots.plot(
        title="Confronto Affitti (Seed: $(simulation_seed))",
        xlabel="Anno",
        ylabel="Valore Indicizzato (Base 100)",
        legend=:topleft
    )
    # Plotta tutte le città reali
    for city_rent_col in names(real_data_df, r"_rent")
        plot!(p_rent, real_data_df.Year, real_data_df[!, city_rent_col], 
              label=replace(city_rent_col, "indexed_" => "", "_rent" => ""), 
              color=:gray, alpha=0.7)
    end
    # Sovrapponi la simulazione
    Plots.plot!(p_rent, sim_years_axis, indexed_sim_rent, 
          label="Simulazione ($(sim_rent_col))", 
          color=:red, linewidth=3)

    # Plot Redditi
    p_income = Plots.plot(
        title="Confronto Redditi (Seed: $(simulation_seed))",
        xlabel="Anno",
        ylabel="Valore Indicizzato (Base 100)",
        legend=:topleft
    )
    # Plotta tutte le città reali
    for city_income_col in names(real_data_df, r"_income")
        plot!(p_income, real_data_df.Year, real_data_df[!, city_income_col], 
              label=replace(city_income_col, "indexed_" => "", "_income" => ""), 
              color=:gray, alpha=0.7)
    end
    # Sovrapponi la simulazione
    Plots.plot!(p_income, sim_years_axis, indexed_sim_income, 
            label="Simulazione ($(sim_income_col))", 
            color=:blue, linewidth=3)
    
    # Mostra i grafici
    display(p_rent)
    display(p_income)
    
    println("--- Grafici di calibrazione generati. ---")
end=#


#=function plot_calibration_trends(
    simulation_df::DataFrame, # Questo è combined_model_metrics_df (step da 0 a fine)
    real_data_df::DataFrame;
    sim_rent_col::Symbol = :median_rent_inner_city,
    sim_income_col::Symbol = :median_income_inner_city,
    steps_per_year::Int = 8,
    moving_avg_window::Int = 5,
    simulation_seed::Int = 0,
    n_prerun_years::Int = 4 # <--- QUESTO È IL PARAMETRO CHIAVE
)
    println("--- Avvio processo di calibrazione visuale ---")
    println("Ipotesi: $(steps_per_year) steps/anno, Finestra Media Mobile: $(moving_avg_window) anni, Anni di Pre-run: $n_prerun_years")

    # --- 1. Aggregazione dati simulazione in "Anni" ---
    # `simulation_df` contiene tutti gli step, dalla termalizzazione inclusa (step 0, 1, ..., n_steps_termalizzazione, n_steps_termalizzazione+1, ...)
    
    # Calcoliamo gli anni concettuali per ogni step della simulazione.
    # L'anno 1 della simulazione dinamica (che è lo step `n_steps_termalizzazione + 1` assoluto)
    # deve corrispondere all'anno `start_year_real - n_prerun_years`.
    
    # Estrai solo i dati dopo la termalizzazione, che sono la base della fase dinamica.
    # Questa parte deve essere precisa per allineare l'anno 1 simulato all'anno concettuale.
    # La fase dinamica inizia allo step `initial_step_for_data_dynamic_phase` (che è n_steps_termalizzazione).
    
    # Assumiamo che `simulation_df` copra l'intero intervallo di simulazione.
    # Calcola l'anno di inizio effettivo della tua serie temporale per la media mobile.
    start_year_real = minimum(real_data_df.Year) # Es. 2011

    # Calcola l'offset per l'anno di inizio delle simulazioni
    # Il primo step della fase dinamica (che è lo step `n_steps_termalizzazione` nel DF se `initial_step_for_data` è 0,
    # o lo step 1 nel DF se `initial_step_for_data` è lo step di inizio dinamica)
    # deve corrispondere all'anno `start_year_real - n_prerun_years`.
    
    # Correggiamo `yearly_sim_data` per avere una colonna `year` che sia l'anno assoluto corretto.
    # Partiamo da `simulation_df` e filtriamo per includere solo gli steps > 0 (o il tuo punto di inizio dinamico)
    # e poi aggreghiamo per anno.

    # Assumiamo che il 'vero' step 1 della dinamica è `n_steps_termalizzazione + 1`
    # E che l'anno reale di inizio di interesse è `start_year_real`.
    # E che la simulazione ha `n_prerun_years` dati prima di `start_year_real`.

    # Filtra il DataFrame per considerare solo gli steps della fase dinamica che contano per il plot
    # partendo da (initial_step_for_data della fase dinamica + 1)
    # E `n_steps_termalizzazione` qui è la fine della fase di termalizzazione, ovvero l'inizio della dinamica
    sim_data_for_plot = filter(row -> row.step >= n_steps_termalizzazione, simulation_df)
    
    # Calcola gli anni assoluti per la simulazione
    # L'anno 0 della fase dinamica è `n_steps_termalizzazione`.
    # Il primo anno da includere è `conceptual_sim_start_year`.
    conceptual_sim_start_year = start_year_real - n_prerun_years # Es: 2011 - 4 = 2007

    sim_data_for_plot.actual_year = floor.(Int, (sim_data_for_plot.step .- n_steps_termalizzazione) ./ steps_per_year) .+ conceptual_sim_start_year

    yearly_sim_data = combine(groupby(sim_data_for_plot, :actual_year),
        sim_rent_col => mean,
        sim_income_col => mean,
        renamecols=false
    )
    sort!(yearly_sim_data, :actual_year)
    println("Dati simulati aggregati in $(nrow(yearly_sim_data)) 'anni' (da $(minimum(yearly_sim_data.actual_year)) a $(maximum(yearly_sim_data.actual_year))).")


    # --- 2. Calcolo Medie Mobili e Indicizzazione ---
    sim_rent_ma = calculate_moving_average(yearly_sim_data[!, sim_rent_col], moving_avg_window)
    sim_income_ma = calculate_moving_average(yearly_sim_data[!, sim_income_col], moving_avg_window)

    if isempty(sim_rent_ma) || isempty(sim_income_ma)
        println("ERRORE: Dati simulati insufficienti per calcolare una media mobile con finestra $(moving_avg_window).")
        println("Anni simulati necessari: $(moving_avg_window), anni disponibili: $(nrow(yearly_sim_data)).")
        return nothing
    end

    indexed_sim_rent = (sim_rent_ma ./ sim_rent_ma[1]) .* 100
    indexed_sim_income = (sim_income_ma ./ sim_income_ma[1]) .* 100

    # L'asse X per la media mobile:
    # Il primo punto della media mobile (sim_rent_ma[1]) corrisponde all'anno
    # `yearly_sim_data.actual_year[1]` + `moving_avg_window` - 1.
    first_ma_year = yearly_sim_data.actual_year[1] + moving_avg_window - 1
    sim_years_axis_ma = first_ma_year : (first_ma_year + length(indexed_sim_rent) - 1)


    # --- 3. Plotting ---
    p_rent = Plots.plot(
        title="Confronto Affitti (Seed: $(simulation_seed))",
        xlabel="Anno",
        ylabel="Valore Indicizzato (Base 100)",
        legend=:topleft
    )
    for city_rent_col in names(real_data_df, r"indexed_.*_rent") # Filtro solo le colonne _rent indicizzate
        Plots.plot!(p_rent, real_data_df.Year, real_data_df[!, city_rent_col],
                      label=replace(replace(string(city_rent_col), "indexed_" => ""), "_rent" => ""),
                      color=:gray, alpha=0.7)
    end
    Plots.plot!(p_rent, sim_years_axis_ma, indexed_sim_rent, # <--- USA `sim_years_axis_ma`
                  label="Simulazione ($(sim_rent_col))",
                  color=:red, linewidth=3)

    p_income = Plots.plot(
        title="Confronto Redditi (Seed: $(simulation_seed))",
        xlabel="Anno",
        ylabel="Valore Indicizzato (Base 100)",
        legend=:topleft
    )
    for city_income_col in names(real_data_df, r"indexed_.*_income") # Filtro solo le colonne _income indicizzate
        Plots.plot!(p_income, real_data_df.Year, real_data_df[!, city_income_col],
                      label=replace(replace(string(city_income_col), "indexed_" => ""), "_income" => ""),
                      color=:gray, alpha=0.7)
    end
    Plots.plot!(p_income, sim_years_axis_ma, indexed_sim_income, # <--- USA `sim_years_axis_ma`
                  label="Simulazione ($(sim_income_col))",
                  color=:blue, linewidth=3)
    
    display(p_rent)
    display(p_income)
    
    println("--- Grafici di calibrazione generati. ---")
    return nothing # La funzione non deve restituire valori se è solo per plotting
end=#


function plot_calibration_trends(
    simulation_df::DataFrame, # Questo è combined_model_metrics_df (step da 0 a fine)
    real_data_df::DataFrame;
    sim_rent_col::Symbol = :median_rent_inner_city,
    sim_income_col::Symbol = :median_income_inner_city,
    steps_per_year::Int = 8,
    moving_avg_window::Int = 5,
    simulation_seed::Int = 0,
    n_prerun_years::Int = 4 # Questo è il parametro chiave aggiunto
)
    println("--- Avvio processo di calibrazione visuale ---")
    println("Ipotesi: $(steps_per_year) steps/anno, Finestra Media Mobile: $(moving_avg_window) anni, Anni di Pre-run: $n_prerun_years")

    # --- 1. Aggregazione dati simulazione in "Anni" ---
    
    local start_step_of_dynamic_phase_in_df = 15
    
    sim_data_for_plot = filter(row -> row.step >= start_step_of_dynamic_phase_in_df, simulation_df)
    
    start_year_real = minimum(real_data_df.Year)

    conceptual_sim_start_year = start_year_real - n_prerun_years

    sim_data_for_plot.actual_year = floor.(Int, (sim_data_for_plot.step .- start_step_of_dynamic_phase_in_df) ./ steps_per_year) .+ conceptual_sim_start_year

    yearly_sim_data = combine(groupby(sim_data_for_plot, :actual_year),
        sim_rent_col => mean,
        sim_income_col => mean,
        renamecols=false
    )
    sort!(yearly_sim_data, :actual_year)
    println("Dati simulati aggregati in $(nrow(yearly_sim_data)) 'anni' (da $(minimum(yearly_sim_data.actual_year)) a $(maximum(yearly_sim_data.actual_year))).")
    
    println("DEBUG: Numero di anni totali in yearly_sim_data: $(nrow(yearly_sim_data))")
    flush(stdout)


    # --- 2. Calcolo Medie Mobili e Indicizzazione ---
    sim_rent_ma = calculate_moving_average(yearly_sim_data[!, sim_rent_col], moving_avg_window)
    sim_income_ma = calculate_moving_average(yearly_sim_data[!, sim_income_col], moving_avg_window)

    if isempty(sim_rent_ma) || isempty(sim_income_ma)
        println("ERRORE: Dati simulati insufficienti per calcolare una media mobile con finestra $(moving_avg_window).")
        println("Anni simulati necessari: $(moving_avg_window), anni disponibili: $(nrow(yearly_sim_data)).")
        return nothing
    end

    indexed_sim_rent = (sim_rent_ma ./ sim_rent_ma[1]) .* 100 #.-100
    indexed_sim_income = (sim_income_ma ./ sim_income_ma[1]) .* 100 #.-100

    first_ma_year = yearly_sim_data.actual_year[1] + moving_avg_window - 1
    sim_years_axis_ma = first_ma_year : (first_ma_year + length(indexed_sim_rent) - 1)

    println("DEBUG: Numero di punti dati dopo media mobile: $(length(indexed_sim_rent))")
    println("DEBUG: Sim MA anii: $(first(sim_years_axis_ma)) to $(last(sim_years_axis_ma))")
    println("DEBUG: Real data anii: $(minimum(real_data_df.Year)) to $(maximum(real_data_df.Year))")
    flush(stdout)

    # --- INIZIO MODIFICA QUI ---
    # Estendi i limiti dell'asse X per vedere eventuali punti oltre i dati reali
    max_year_simulated = last(sim_years_axis_ma) # L'ultimo anno effettivo che la simulazione copre con MA
    # Imposta il limite superiore dell'asse X al massimo tra l'ultimo anno reale e l'ultimo anno simulato, più un piccolo margine
    x_upper_limit_plot = max(maximum(real_data_df.Year), max_year_simulated) + 1 # Un anno in più per vedere il punto
    x_lower_limit_plot = minimum(real_data_df.Year) # L'inizio rimane lo stesso
    x_axis_limits = (x_lower_limit_plot, x_upper_limit_plot)
    # --- FINE MODIFICA QUI ---
    #=
    p_rent = Plots.plot(
        #title="Confronto Affitti (Seed: $(simulation_seed))",
        xlabel="Year",
        ylabel="Indexed median rent",
        legend=false,
        xlims=x_axis_limits # <--- ORA USA I NUOVI LIMITI ESTESI
    )
    for city_rent_col in names(real_data_df, r"indexed_.*_rent")
        Plots.plot!(p_rent, real_data_df.Year, real_data_df[!, city_rent_col],
                      #label=replace(replace(string(city_rent_col), "indexed_" => ""), "_rent" => ""),
                      color=:gray, alpha=0.7)
    end
    Plots.plot!(p_rent, sim_years_axis_ma, indexed_sim_rent,
                  label="seed ($(sim_rent_col))",
                  color=:red, linewidth=3,
                  marker=:circle, markersize=4)

    p_income = Plots.plot(
        #title="Confronto Redditi (Seed: $(simulation_seed))",
        xlabel="Year",
        ylabel="Indexed median income",
        legend=false,
        xlims=x_axis_limits # <--- ORA USA I NUOVI LIMITI ESTESI
    )
    for city_income_col in names(real_data_df, r"indexed_.*_income")
        Plots.plot!(p_income, real_data_df.Year, real_data_df[!, city_income_col],
                      #label=replace(replace(string(city_income_col),"indexed_" => ""), "_income" => ""),
                      color=:gray, alpha=0.7)
    end
    Plots.plot!(p_income, sim_years_axis_ma, indexed_sim_income,
                  label="seed ($(sim_income_col))",
                  color=:blue, linewidth=3,
                  marker=:circle, markersize=4)=#

    # Grafico per gli affitti
    p_rent = Plots.plot(
        xlabel="Year",
        ylabel="Indexed Median Rent",
        legend=:topleft, # <-- 1. Riattiva la legenda
        xlims=x_axis_limits
    )
    # 2. Plotta i dati reali senza etichetta
    for city_rent_col in names(real_data_df, r"indexed_.*_rent")
        Plots.plot!(p_rent, real_data_df.Year, real_data_df[!, city_rent_col],
                      label="", color=:gray, alpha=0.7, linestyle=:dash)
    end
    # 3. Plotta la simulazione CON l'etichetta
    Plots.plot!(p_rent, sim_years_axis_ma, indexed_sim_rent,
                  label="Simulated Data (Seed $(simulation_seed))", 
                  color=:red, linewidth=3,
                  marker=:circle, markersize=4)
    # 4. Aggiungi la linea "finta" per la legenda dei dati reali
    Plots.plot!(p_rent, [], [], color=:gray, linestyle=:dash, label="Empirical Data")


    # Grafico per i redditi (stessa logica)
    p_income = Plots.plot(
        xlabel="Year",
        ylabel="Indexed Median Income",
        legend=:topleft, # <-- 1. Riattiva la legenda
        xlims=x_axis_limits
    )
    for city_income_col in names(real_data_df, r"indexed_.*_income")
        Plots.plot!(p_income, real_data_df.Year, real_data_df[!, city_income_col],
                      label="", color=:gray, alpha=0.7, linestyle=:dash)
    end
    Plots.plot!(p_income, sim_years_axis_ma, indexed_sim_income,
                  label="Simulated Data (Seed $(simulation_seed))", 
                  color=:blue, linewidth=3,
                  marker=:circle, markersize=4)
    Plots.plot!(p_income, [], [], color=:gray, linestyle=:dash, label="Empirical Data")
    
    
    display(p_rent)
    display(p_income)
    
    println("--- Grafici di calibrazione generati. ---")
    return nothing

end

function run_and_plot_final_scenario(
    seeds::Vector{Int}, 
    real_data_df::DataFrame;
    # --- Parametri della funzione ---
    title_suffix::String = "Confronto Calibrato",
    steps_per_year::Int = 12,
    n_steps_term::Int = 15,
    grid_dims::Tuple{Int, Int} = (40, 40),
    num_agents::Int = 800,
    w_poor::Float64
)
    println("\n--- Esecuzione Scenario Finale (Colori Distinti): $title_suffix ---")

    # --- 1. IMPOSTAZIONI ---
    n_prerun_years = 4 
    n_interest_years = 10
    n_steps_total_dinamica = (n_prerun_years + n_interest_years) * steps_per_year
    
    # MODIFICA: Definiamo una palette di colori per le simulazioni
    sim_colors = [:red, :blue, :green, :purple, :orange, :cyan]

    # --- 2. PLOT DATI REALI ---
    p_rent = Plots.plot(title="Trend Affitto Mediano - $title_suffix", xlabel="Anno", ylabel="Indice (Anno Base 2011 = 100)")
    p_income = Plots.plot(title="Trend Reddito Mediano - $title_suffix", xlabel="Anno", ylabel="Indice (Anno Base 2011 = 100)")

    citta_da_plottare = [:Brooklyn, :DC, :Houston, :Austin, :Philly]
    for city in citta_da_plottare
        rent_col = Symbol("indexed_$(lowercase(string(city)))_rent")
        income_col = Symbol("indexed_$(lowercase(string(city)))_income")
        Plots.plot!(p_rent, real_data_df.Year, real_data_df[!, rent_col], label=string(city), linestyle=:dash, lw=2)
        Plots.plot!(p_income, real_data_df.Year, real_data_df[!, income_col], label="", linestyle=:dash, lw=2)
    end
    
    # --- 3. LOOP DI SIMULAZIONE ---
    # MODIFICA: Usiamo enumerate per avere un indice 'i' per i colori
    for (i, seed) in enumerate(seeds)
        println("  -> Esecuzione simulazione per il Seed: $seed")
        
        # Esecuzione...
        params = (dims=grid_dims, N_agents=num_agents, income_distribution=create_mixed_distribution(), w_poor=w_poor, seed=seed)
        initial_model = initialize_model(; params...)
        _, _, thermalized_model = run_simulation_phase_and_collect_data(initial_model, n_steps_term, agent_step!, model_step_termalizzazione!, 0, "Termalizzazione")
        _, model_metrics_df, _ = run_simulation_phase_and_collect_data(thermalized_model, n_steps_total_dinamica, agent_step!, model_step_dinamica!, n_steps_term, "Dinamica Lunga")
        
        # Analisi robusta...
        raw_rent = model_metrics_df.median_inner_city_rent
        n_anni_effettivi = floor(Int, length(raw_rent) / steps_per_year)
        if n_anni_effettivi < 5
            println("    ATTENZIONE: Dati insufficienti per Seed $seed. Saltato.")
            continue
        end
        
        # Processamento...
        raw_income = model_metrics_df.median_income_inner_city
        sim_annual_rent = [mean(raw_rent[((i-1)*steps_per_year+1):(i*steps_per_year)]) for i in 1:n_anni_effettivi]
        sim_annual_income = [mean(raw_income[((i-1)*steps_per_year+1):(i*steps_per_year)]) for i in 1:n_anni_effettivi]
        sim_5yr_rent_raw = rollmean(sim_annual_rent, 5)
        sim_5yr_income_raw = rollmean(sim_annual_income, 5)
        indexed_sim_rent = (sim_5yr_rent_raw ./ sim_5yr_rent_raw[1]) .* 100
        indexed_sim_income = (sim_5yr_income_raw ./ sim_5yr_income_raw[1]) .* 100
        anni_da_plottare_effettivi = 2011:(2011 + length(indexed_sim_rent) - 1)
        
        # MODIFICA: Assegnamo un colore dalla palette
        color_sim = sim_colors[mod1(i, length(sim_colors))] # `mod1` ricicla i colori se i seed sono più della palette

        Plots.plot!(p_rent, anni_da_plottare_effettivi, indexed_sim_rent, label="Sim (Seed $seed)", lw=2.5, marker=:circle, markersize=3, color=color_sim)
        Plots.plot!(p_income, anni_da_plottare_effettivi, indexed_sim_income, label="", lw=2.5, marker=:circle, markersize=3, color=color_sim)
    end

    # --- 4. VISUALIZZAZIONE FINALE ---
    final_plot = Plots.plot(p_rent, p_income, layout=(1,2), size=(1500, 600), plot_title="Risultati Scenario: $title_suffix", legend=:outerright)
    display(final_plot)
end


function calculate_rmse(simulated::Vector{<:Real}, real::Vector{<:Real})
    len = min(length(simulated), length(real))
    if len == 0; return Inf; end
    return sqrt(mean((simulated[1:len] .- real[1:len]).^2))
end


#=Metrica 1: calculate_rmse (Root Mean Square Error)
Nome Completo: RMSE - Root Mean Square Error (in italiano, "Radice dell'Errore Quadratico Medio").
In Breve: Misura la "distanza media" tra i valori della tua simulazione e i valori reali, punto per punto.
A quale domanda risponde: “In media, di quanto si sono sbagliati i valori della mia simulazione rispetto a quelli veri?”
Come Funziona (Passo per Passo)
Immagina di avere la linea dei dati reali e la linea della tua simulazione su un grafico.

Errore (Error): Per ogni anno (ogni punto del grafico), calcola la differenza verticale tra il valore simulato e il valore reale. Questo è l' "errore" per quell'anno. A volte sarà positivo (hai sovrastimato), a volte negativo (hai sottostimato).

simulated[1:len] .- real[1:len]
Quadratico (Square): Eleva al quadrato ogni singolo errore. Questo ha due effetti importantissimi:

Rende tutto positivo: Non ci importa più se l'errore era in eccesso o in difetto, ma solo la sua grandezza.
Punisce di più gli errori grandi: Un errore di 2 diventa 4, ma un errore di 10 diventa 100. Questo significa che l'RMSE è molto sensibile ai grandi "strafalcioni" della simulazione.
(...).^2
Medio (Mean): Fai la media di tutti questi errori quadratici. Ottieni un singolo numero che rappresenta l'errore quadratico medio su tutta la simulazione.

mean(...)
Radice (Root): Estrai la radice quadrata del risultato. Questo passaggio serve principalmente a riportare la metrica all'unità di misura originale. Se stavi misurando l'errore in "dollari", dopo averlo elevato al quadrato era in "dollari quadrati". La radice lo riporta in "dollari", rendendolo più interpretabile.

sqrt(...)
Punti di Forza e Deboli
Forza: È una metrica standard, molto usata e facile da capire. Penalizzando molto gli errori grandi, ti spinge a trovare modelli che evitano deviazioni estreme.
Debolezza: Non sa nulla della "forma" o della dinamica del trend. Immagina due simulazioni: una è una linea piatta che è costantemente 10 unità sopra la linea reale. L'altra segue perfettamente la forma della linea reale, ma è costantemente 10 unità sotto. Per l'RMSE, queste due simulazioni sono identicamente sbagliate, anche se la seconda è chiaramente un modello migliore dal punto di vista qualitativo.
Nel Tuo Contesto: L'RMSE ti dice se, in media, i tuoi valori di affitto/reddito indicizzati sono stati quantitativamente vicini a quelli reali. È una misura di accuratezza numerica.
=#



"""
Calcola la distanza DTW (Dynamic Time Warping) tra due serie.
Richiede il pacchetto DynamicAxisWarping.
"""
function calculate_shape_distance(simulated::Vector{<:Real}, real::Vector{<:Real})
    len = min(length(simulated), length(real))
    if len < 2 # Abbiamo bisogno di almeno due punti per calcolare una differenza
        return Inf
    end
    
    # Calcola la serie delle differenze (crescita anno per anno)
    sim_growth = diff(simulated[1:len])
    real_growth = diff(real[1:len])
    
    # Restituisce l'RMSE di queste serie di crescita
    return calculate_rmse(sim_growth, real_growth)
end

#=Metrica 2: calculate_shape_distance (Distanza di Forma)
Nome Descrittivo: Potremmo chiamarla "Distanza della Dinamica di Crescita".
In Breve: Non confronta i valori assoluti, ma la somiglianza nel "ritmo di crescita" tra la simulazione e i dati reali.
A quale domanda risponde: “La mia simulazione accelera e decelera nei momenti giusti, proprio come nella realtà, anche se i suoi valori assoluti sono un po' sballati?”
Come Funziona (Passo per Passo)
Questa metrica è molto intelligente. Ignora i valori originali e si concentra su qualcos'altro: la crescita anno per anno.

Calcola la Crescita (Growth): Invece di usare le serie di dati originali (es. [100, 105, 115]), calcola la serie delle differenze tra un anno e l'altro.

La serie [100, 105, 115] diventa [5, 10]. Questo nuovo vettore rappresenta la "crescita" o la "velocità" del fenomeno.
sim_growth = diff(simulated[1:len])
real_growth = diff(real[1:len])
Calcola l'RMSE... sulla Crescita! A questo punto, fa una cosa semplice ma potente: calcola l'RMSE non tra i valori originali, ma tra i due vettori di crescita.

calculate_rmse(sim_growth, real_growth)
=#

function find_best_seeds(
    seeds_to_test::Vector{Int},
    real_data_df::DataFrame;
    # Parametri fissi
    steps_per_year::Int = 12,
    n_steps_term::Int = 15,
    grid_dims::Tuple{Int, Int} = (40, 40),
    num_agents::Int = 800
)
    println("--- Avvio Ricerca dei Seed Ottimali (con RMSE e Distanza di Forma) ---")
    
    # DataFrame per i risultati con le nuove metriche
    results = DataFrame(
        seed=Int[], 
        avg_rmse_rent=Float64[], avg_rmse_income=Float64[],
        avg_shape_dist_rent=Float64[], avg_shape_dist_income=Float64[] # Sostituisce DTW
    )
    
    citta_di_riferimento = [:Brooklyn, :DC, :Houston, :Austin, :Philly]

    for seed in seeds_to_test
        println("\n=> Test con Seed = $seed")
        
        # --- Esecuzione e Processamento (invariato) ---
        w_poor = probs(create_mixed_distribution().prior)[1]
        n_prerun_years = 4 
        n_interest_years = 10
        n_steps_total_dinamica = (n_prerun_years + n_interest_years) * steps_per_year
        params = (dims=grid_dims, N_agents=num_agents, income_distribution=create_mixed_distribution(), w_poor=w_poor, seed=seed)
        initial_model = initialize_model(; params...)
        # ... (il resto della simulazione è identico)
        _, _, thermalized_model = run_simulation_phase_and_collect_data(initial_model, n_steps_term, agent_step!, model_step_termalizzazione!, 0, "Termalizzazione")
        _, model_metrics_df, _ = run_simulation_phase_and_collect_data(thermalized_model, n_steps_total_dinamica, agent_step!, model_step_dinamica!, n_steps_term, "Dinamica Lunga")
        raw_rent = model_metrics_df.median_inner_city_rent
        raw_income = model_metrics_df.median_income_inner_city
        n_anni_effettivi = floor(Int, length(raw_rent) / steps_per_year)
        if n_anni_effettivi < 5
            push!(results, [seed, Inf, Inf, Inf, Inf])
            continue
        end
        sim_annual_rent = [mean(raw_rent[((i-1)*steps_per_year+1):(i*steps_per_year)]) for i in 1:n_anni_effettivi]
        sim_annual_income = [mean(raw_income[((i-1)*steps_per_year+1):(i*steps_per_year)]) for i in 1:n_anni_effettivi]
        sim_5yr_rent_raw = rollmean(sim_annual_rent, 5)
        sim_5yr_income_raw = rollmean(sim_annual_income, 5)
        indexed_sim_rent = (sim_5yr_rent_raw ./ sim_5yr_rent_raw[1]) .* 100
        indexed_sim_income = (sim_5yr_income_raw ./ sim_5yr_income_raw[1]) .* 100

        # --- Calcolo di entrambe le metriche per ogni città ---
        list_rmse_r, list_rmse_i = Float64[], Float64[]
        list_shape_r, list_shape_i = Float64[], Float64[]

        for city in citta_di_riferimento
            real_rent_series = real_data_df[!, Symbol("indexed_$(lowercase(string(city)))_rent")]
            real_income_series = real_data_df[!, Symbol("indexed_$(lowercase(string(city)))_income")]
            
            # Calcola e salva entrambe le metriche
            push!(list_rmse_r, calculate_rmse(indexed_sim_rent, real_rent_series))
            push!(list_rmse_i, calculate_rmse(indexed_sim_income, real_income_series))
            push!(list_shape_r, calculate_shape_distance(indexed_sim_rent, real_rent_series))
            push!(list_shape_i, calculate_shape_distance(indexed_sim_income, real_income_series))
        end
        
        # Media delle metriche e salvataggio
        push!(results, [seed, mean(list_rmse_r), mean(list_rmse_i), mean(list_shape_r), mean(list_shape_i)])
    end

    # --- Analisi Finale ---
    println("\n--- Classifica dei Seed Testati ---")
    results.combined_rmse = results.avg_rmse_rent .+ results.avg_rmse_income
    results.combined_shape = results.avg_shape_dist_rent .+ results.avg_shape_dist_income
    
    println("\n=> Classifica per RMSE (Accuratezza Quantitativa):")
    display(sort(results, :combined_rmse)[!, [:seed, :combined_rmse]])
    
    println("\n=> Classifica per Distanza di Forma (Dinamica di Crescita Simile):")
    display(sort(results, :combined_shape)[!, [:seed, :combined_shape]])
    
    return results
end

#=
function calibrate_and_plot_best_seeds(
    seeds_to_test::Vector{Int},
    real_data_df::DataFrame;
    # --- Nuovi parametri per il controllo ---
    top_n_to_plot::Int = 3, # Quanti dei migliori seed vuoi visualizzare?
    sort_by_metric::Symbol = :rmse, # Scegli se ordinare per :rmse o :shape
    # --- Parametri di simulazione ---
    steps_per_year::Int = 8,
    n_steps_term::Int = 15,
    grid_dims::Tuple{Int, Int} = (40, 40),
    num_agents::Int = 800
)
    println("--- Avvio Esperimento Unificato: Calibrazione e Plot ---")
    
    # MODIFICA: Il DataFrame ora salva anche le serie di dati generate
    results = DataFrame(
        seed=Int[], 
        avg_rmse_rent=Float64[], avg_rmse_income=Float64[],
        avg_shape_dist_rent=Float64[], avg_shape_dist_income=Float64[],
        rent_series=Vector[], # Salveremo qui il vettore di dati dell'affitto
        income_series=Vector[] # e qui quello del reddito
        
    )
    
    citta_di_riferimento = [:Brooklyn, :DC, :Houston, :Austin, :Philly]

    # --- 1. FASE DI SIMULAZIONE E RACCOLTA DATI ---
    for seed in seeds_to_test
        println("  -> Esecuzione e analisi per il Seed: $seed")
        
        # Esecuzione e Processamento...
        w_poor = probs(create_mixed_distribution().prior)[1]
        n_prerun_years = 4 
        n_interest_years = 10
        n_steps_total_dinamica = (n_prerun_years + n_interest_years) * steps_per_year
        params = (dims=grid_dims, N_agents=num_agents, income_distribution=create_mixed_distribution(), w_poor=w_poor, seed=seed)
        # ... (tutta la logica di esecuzione è identica a prima) ...
        initial_model = initialize_model(; params...)
        _, _, thermalized_model = run_simulation_phase_and_collect_data(initial_model, n_steps_term, agent_step!, model_step_termalizzazione!, 0, "Termalizzazione")
        _, model_metrics_df, _ = run_simulation_phase_and_collect_data(thermalized_model, n_steps_total_dinamica, agent_step!, model_step_dinamica!, n_steps_term, "Dinamica Lunga")
        raw_rent = model_metrics_df.median_inner_city_rent
        raw_income = model_metrics_df.median_income_inner_city
        n_anni_effettivi = floor(Int, length(raw_rent) / steps_per_year)
        if n_anni_effettivi < 5
            push!(results, [seed, Inf, Inf, Inf, Inf, [], []])
            continue
        end
        sim_annual_rent = [mean(raw_rent[((i-1)*steps_per_year+1):(i*steps_per_year)]) for i in 1:n_anni_effettivi]
        sim_annual_income = [mean(raw_income[((i-1)*steps_per_year+1):(i*steps_per_year)]) for i in 1:n_anni_effettivi]
        sim_5yr_rent_raw = rollmean(sim_annual_rent, 5)
        sim_5yr_income_raw = rollmean(sim_annual_income, 5)
        indexed_sim_rent = (sim_5yr_rent_raw ./ sim_5yr_rent_raw[1]) .* 100
        indexed_sim_income = (sim_5yr_income_raw ./ sim_5yr_income_raw[1]) .* 100

        # Calcolo delle metriche...
        list_rmse_r, list_rmse_i = Float64[], Float64[]
        list_shape_r, list_shape_i = Float64[], Float64[]
        for city in citta_di_riferimento
            real_rent_series = real_data_df[!, Symbol("indexed_$(lowercase(string(city)))_rent")]
            real_income_series = real_data_df[!, Symbol("indexed_$(lowercase(string(city)))_income")]
            push!(list_rmse_r, calculate_rmse(indexed_sim_rent, real_rent_series))
            push!(list_rmse_i, calculate_rmse(indexed_sim_income, real_income_series))
            push!(list_shape_r, calculate_shape_distance(indexed_sim_rent, real_rent_series))
            push!(list_shape_i, calculate_shape_distance(indexed_sim_income, real_income_series))
        end
        
        # MODIFICA: Salviamo nel DataFrame anche le serie di dati calcolate
        push!(results, [seed, mean(list_rmse_r), mean(list_rmse_i), mean(list_shape_r), mean(list_shape_i), indexed_sim_rent, indexed_sim_income])
    end

    # --- 2. FASE DI ANALISI E PLOTTING (USA I DATI SALVATI) ---
    println("\n--- Classifica dei Seed Testati ---")
    results.combined_rmse = results.avg_rmse_rent .+ results.avg_rmse_income
    results.combined_shape = results.avg_shape_dist_rent .+ results.avg_shape_dist_income
    
    # Determina la colonna per l'ordinamento
    sort_col = sort_by_metric == :rmse ? :combined_rmse : :combined_shape
    sorted_results = sort(results, sort_col)
    
    println("\n=> Classifica completa ordinata per $(sort_col):")
    display(sorted_results[!, [:seed, :combined_rmse, :combined_shape]])

    # Ora il plot viene generato automaticamente per i migliori N seed
    println("\n--- Generazione Grafico per i Migliori $top_n_to_plot Seed ---")
    p_rent = Plots.plot(title="Confronto Migliori Seed (Affitto)", xlabel="Anno", ylabel="Indice (Anno Base 2011 = 100)")
    p_income = Plots.plot(title="Confronto Migliori Seed (Reddito)", xlabel="Anno", ylabel="Indice (Anno Base 2011 = 100)")

    # Plot dati reali
    for city in citta_di_riferimento
        rent_col = Symbol("indexed_$(lowercase(string(city)))_rent")
        Plots.plot!(p_rent, real_data_df.Year, real_data_df[!, rent_col], label=string(city), linestyle=:dash, lw=2)
        Plots.plot!(p_income, real_data_df.Year, real_data_df[!, rent_col], label="", linestyle=:dash, lw=2)
    end
    
    # Plot dei migliori N seed, usando i dati salvati
    sim_colors = [:red, :blue, :green, :purple, :orange]
    for (i, row) in enumerate(eachrow(first(sorted_results, top_n_to_plot)))
        anni_da_plottare_effettivi = 2011:(2011 + length(row.rent_series) - 1)
        color = sim_colors[mod1(i, length(sim_colors))]
        
        Plots.plot!(p_rent, anni_da_plottare_effettivi, row.rent_series, label="Seed $(row.seed)", lw=2.5, marker=:circle, markersize=3, color=color)
        Plots.plot!(p_income, anni_da_plottare_effettivi, row.income_series, label="", lw=2.5, marker=:circle, markersize=3, color=color)
    end

    final_plot = Plots.plot(p_rent, p_income, layout=(1,2), size=(1500, 600), legend=:outerright)
    display(final_plot)
    
    return sorted_results
end=#


function calibrate_and_plot_best_seeds(
    seeds_to_test::Vector{Int},
    real_data_df::DataFrame;
    # --- Nuovi parametri per il controllo ---
    top_n_to_plot::Int = 3, # Quanti dei migliori seed vuoi visualizzare?
    sort_by_metric::Symbol = :rmse, # Scegli se ordinare per :rmse o :shape
    # --- Parametri di simulazione ---
    steps_per_year::Int = 8,
    n_steps_term::Int = 15,
    grid_dims::Tuple{Int, Int} = (40, 40),
    num_agents::Int = 800
)
    println("--- Avvio Esperimento Unificato: Calibrazione e Plot ---")
    
    # MODIFICA: Il DataFrame ora salva anche le serie di dati generate
    results = DataFrame(
        seed=Int[], 
        avg_rmse_rent=Float64[], avg_rmse_income=Float64[],
        avg_shape_dist_rent=Float64[], avg_shape_dist_income=Float64[],
        rent_series=Vector[], # Salveremo qui il vettore di dati dell'affitto
        income_series=Vector[], # e qui quello del reddito
        final_rent_index=Float64[], # <-- NUOVO
        final_income_index=Float64[] # <-- NUOVO
    )
    
    citta_di_riferimento = [:Brooklyn, :DC, :Houston, :Austin, :Philly]

    # --- 1. FASE DI SIMULAZIONE E RACCOLTA DATI ---
    for seed in seeds_to_test
        println("  -> Esecuzione e analisi per il Seed: $seed")
        
        # Esecuzione e Processamento...
        w_poor = probs(create_mixed_distribution().prior)[1]
        n_prerun_years = 4 
        n_interest_years = 10
        n_steps_total_dinamica = (n_prerun_years + n_interest_years) * steps_per_year
        params = (dims=grid_dims, N_agents=num_agents, income_distribution=create_mixed_distribution(), w_poor=w_poor, seed=seed)
        # ... (tutta la logica di esecuzione è identica a prima) ...
        initial_model = initialize_model(; params...)
        _, _, thermalized_model = run_simulation_phase_and_collect_data(initial_model, n_steps_term, agent_step!, model_step_termalizzazione!, 0, "Termalizzazione")
        _, model_metrics_df, _ = run_simulation_phase_and_collect_data(thermalized_model, n_steps_total_dinamica, agent_step!, model_step_dinamica!, n_steps_term, "Dinamica Lunga")
        raw_rent = model_metrics_df.median_inner_city_rent
        raw_income = model_metrics_df.median_income_inner_city
        n_anni_effettivi = floor(Int, length(raw_rent) / steps_per_year)
        if n_anni_effettivi < 5
            push!(results, [seed, Inf, Inf, Inf, Inf, [], []])
            continue
        end
        sim_annual_rent = [mean(raw_rent[((i-1)*steps_per_year+1):(i*steps_per_year)]) for i in 1:n_anni_effettivi]
        sim_annual_income = [mean(raw_income[((i-1)*steps_per_year+1):(i*steps_per_year)]) for i in 1:n_anni_effettivi]
        sim_5yr_rent_raw = rollmean(sim_annual_rent, 5)
        sim_5yr_income_raw = rollmean(sim_annual_income, 5)
        indexed_sim_rent = (sim_5yr_rent_raw ./ sim_5yr_rent_raw[1]) .* 100
        indexed_sim_income = (sim_5yr_income_raw ./ sim_5yr_income_raw[1]) .* 100

        final_rent = isempty(indexed_sim_rent) ? NaN : last(indexed_sim_rent)
        final_income = isempty(indexed_sim_income) ? NaN : last(indexed_sim_income)



        # Calcolo delle metriche...
        list_rmse_r, list_rmse_i = Float64[], Float64[]
        list_shape_r, list_shape_i = Float64[], Float64[]
        for city in citta_di_riferimento
            real_rent_series = real_data_df[!, Symbol("indexed_$(lowercase(string(city)))_rent")]
            real_income_series = real_data_df[!, Symbol("indexed_$(lowercase(string(city)))_income")]
            push!(list_rmse_r, calculate_rmse(indexed_sim_rent, real_rent_series))
            push!(list_rmse_i, calculate_rmse(indexed_sim_income, real_income_series))
            push!(list_shape_r, calculate_shape_distance(indexed_sim_rent, real_rent_series))
            push!(list_shape_i, calculate_shape_distance(indexed_sim_income, real_income_series))
        end
        
        # MODIFICA: Salviamo nel DataFrame anche le serie di dati calcolate
        push!(results, [
            seed, mean(list_rmse_r), mean(list_rmse_i), 
            mean(list_shape_r), mean(list_shape_i), 
            indexed_sim_rent, indexed_sim_income,
            final_rent, final_income # <-- NUOVO
        ])
    end

    #--- 2. FASE DI ANALISI E PLOTTING (USA I DATI SALVATI) ---
    #=
    println("\n--- Classifica dei Seed Testati ---")
    results.combined_rmse = results.avg_rmse_rent .+ results.avg_rmse_income
    results.combined_shape = results.avg_shape_dist_rent .+ results.avg_shape_dist_income
    
    # Determina la colonna per l'ordinamento
    sort_col = sort_by_metric == :rmse ? :combined_rmse : :combined_shape
    sorted_results = sort(results, sort_col)
    
    println("\n=> Classifica completa ordinata per $(sort_col):")
    display(sorted_results[!, [:seed, :combined_rmse, :combined_shape]])

    # Ora il plot viene generato automaticamente per i migliori N seed
    println("\n--- Generazione Grafico per i Migliori $top_n_to_plot Seed ---")
    p_rent = Plots.plot(title="Confronto Migliori Seed (Affitto)", xlabel="Anno", ylabel="Indice (Anno Base 2011 = 100)")
    p_income = Plots.plot(title="Confronto Migliori Seed (Reddito)", xlabel="Anno", ylabel="Indice (Anno Base 2011 = 100)")

    # Plot dati reali
    for city in citta_di_riferimento
        rent_col = Symbol("indexed_$(lowercase(string(city)))_rent")
        Plots.plot!(p_rent, real_data_df.Year, real_data_df[!, rent_col], label=string(city), linestyle=:dash, lw=2)
        Plots.plot!(p_income, real_data_df.Year, real_data_df[!, rent_col], label="", linestyle=:dash, lw=2)
    end
    
    # Plot dei migliori N seed, usando i dati salvati
    sim_colors = [:red, :blue, :green, :purple, :orange]
    for (i, row) in enumerate(eachrow(first(sorted_results, top_n_to_plot)))
        anni_da_plottare_effettivi = 2011:(2011 + length(row.rent_series) - 1)
        color = sim_colors[mod1(i, length(sim_colors))]
        
        Plots.plot!(p_rent, anni_da_plottare_effettivi, row.rent_series, label="Seed $(row.seed)", lw=2.5, marker=:circle, markersize=3, color=color)
        Plots.plot!(p_income, anni_da_plottare_effettivi, row.income_series, label="", lw=2.5, marker=:circle, markersize=3, color=color)
    end

    final_plot = Plots.plot(p_rent, p_income, layout=(1,2), size=(1500, 600), legend=:outerright)
    display(final_plot)   
=#

    results.combined_rmse = results.avg_rmse_rent .+ results.avg_rmse_income
    results.combined_shape = results.avg_shape_dist_rent .+ results.avg_shape_dist_income


    return results
end


function calibrate_and_plot_best_seeds_with_plot(
    seeds_to_test::Vector{Int},
    real_data_df::DataFrame;
    # --- Nuovi parametri per il controllo ---
    top_n_to_plot::Int = 3, # Quanti dei migliori seed vuoi visualizzare?
    sort_by_metric::Symbol = :rmse, # Scegli se ordinare per :rmse o :shape
    # --- Parametri di simulazione ---
    steps_per_year::Int = 8,
    n_steps_term::Int = 15,
    grid_dims::Tuple{Int, Int} = (40, 40),
    num_agents::Int = 800
)
    println("--- Avvio Esperimento Unificato: Calibrazione e Plot ---")
    
    # MODIFICA: Il DataFrame ora salva anche le serie di dati generate
    results = DataFrame(
        seed=Int[], 
        avg_rmse_rent=Float64[], avg_rmse_income=Float64[],
        avg_shape_dist_rent=Float64[], avg_shape_dist_income=Float64[],
        rent_series=Vector[], # Salveremo qui il vettore di dati dell'affitto
        income_series=Vector[], # e qui quello del reddito
        final_rent_index=Float64[], # <-- NUOVO
        final_income_index=Float64[] # <-- NUOVO
    )
    
    citta_di_riferimento = [:Brooklyn, :DC, :Houston, :Austin, :Philly]

    # --- 1. FASE DI SIMULAZIONE E RACCOLTA DATI ---
    # --- 1. FASE DI SIMULAZIONE E RACCOLTA DATI ---
    for seed in seeds_to_test
        println("    -> Esecuzione e analisi per il Seed: $seed")
        
        w_poor = probs(create_mixed_distribution().prior)[1]
        n_prerun_years = 4 
        n_interest_years = 10
        n_steps_total_dinamica = (n_prerun_years + n_interest_years) * steps_per_year
        params = (dims=grid_dims, N_agents=num_agents, income_distribution=create_mixed_distribution(), w_poor=w_poor, seed=seed)
        
        initial_model = initialize_model(; params...)

        # MODIFICA 1: Cattura tutti e 4 i valori restituiti, incluso il modello aggiornato
        _, _, _, thermalized_model_object = run_simulation_phase_and_collect_data(
            initial_model,
            n_steps_term,
            agent_step!,
            model_step_termalizzazione!,
            0,
            "Termalizzazione"
        )

        # MODIFICA 2: Cattura tutti e 4 i valori restituiti, incluso il modello finale
        # E passa l'output del modello della fase precedente come input per questa fase.
        _, model_metrics_df, _, final_dynamic_model_object = run_simulation_phase_and_collect_data(
            thermalized_model_object, # <--- Passa il modello aggiornato dalla fase di termalizzazione
            n_steps_total_dinamica,
            agent_step!,
            model_step_dinamica!,
            n_steps_term,
            "Dinamica Lunga"
        )
        
        raw_rent = model_metrics_df.median_inner_city_rent
        raw_income = model_metrics_df.median_income_inner_city
        n_anni_effettivi = floor(Int, length(raw_rent) / steps_per_year)
        if n_anni_effettivi < 5
            push!(results, [seed, Inf, Inf, Inf, Inf, [], [], NaN, NaN]) # Aggiungi NaN per final_rent_index, final_income_index
            continue
        end
        sim_annual_rent = [mean(raw_rent[((i-1)*steps_per_year+1):(i*steps_per_year)]) for i in 1:n_anni_effettivi]
        sim_annual_income = [mean(raw_income[((i-1)*steps_per_year+1):(i*steps_per_year)]) for i in 1:n_anni_effettivi]
        sim_5yr_rent_raw = rollmean(sim_annual_rent, 5)
        sim_5yr_income_raw = rollmean(sim_annual_income, 5)
        indexed_sim_rent = (sim_5yr_rent_raw ./ sim_5yr_rent_raw[1]) .* 100
        indexed_sim_income = (sim_5yr_income_raw ./ sim_5yr_income_raw[1]) .* 100

        final_rent = isempty(indexed_sim_rent) ? NaN : last(indexed_sim_rent)
        final_income = isempty(indexed_sim_income) ? NaN : last(indexed_sim_income)

        list_rmse_r, list_rmse_i = Float64[], Float64[]
        list_shape_r, list_shape_i = Float64[], Float64[]
        for city in citta_di_riferimento
            real_rent_series = real_data_df[!, Symbol("indexed_$(lowercase(string(city)))_rent")]
            real_income_series = real_data_df[!, Symbol("indexed_$(lowercase(string(city)))_income")]
            push!(list_rmse_r, calculate_rmse(indexed_sim_rent, real_rent_series))
            push!(list_rmse_i, calculate_rmse(indexed_sim_income, real_income_series))
            push!(list_shape_r, calculate_shape_distance(indexed_sim_rent, real_rent_series))
            push!(list_shape_i, calculate_shape_distance(indexed_sim_income, real_income_series))
        end
        
        push!(results, [
            seed, mean(list_rmse_r), mean(list_rmse_i), 
            mean(list_shape_r), mean(list_shape_i), 
            indexed_sim_rent, indexed_sim_income,
            final_rent, final_income
        ])
    end

    #--- 2. FASE DI ANALISI E PLOTTING (USA I DATI SALVATI) ---
    
    println("\n--- Classifica dei Seed Testati ---")
    results.combined_rmse = results.avg_rmse_rent .+ results.avg_rmse_income
    results.combined_shape = results.avg_shape_dist_rent .+ results.avg_shape_dist_income
    
    # Determina la colonna per l'ordinamento
    sort_col = sort_by_metric == :rmse ? :combined_rmse : :combined_shape
    sorted_results = sort(results, sort_col)
    
    println("\n=> Classifica completa ordinata per $(sort_col):")
    display(sorted_results[!, [:seed, :combined_rmse, :combined_shape]])

    # Ora il plot viene generato automaticamente per i migliori N seed
    println("\n--- Generazione Grafico per i Migliori $top_n_to_plot Seed ---")
    p_rent = Plots.plot(title="Confronto Migliori Seed (Affitto)", xlabel="Anno", ylabel="Indice (Anno Base 2011 = 100)")
    p_income = Plots.plot(title="Confronto Migliori Seed (Reddito)", xlabel="Anno", ylabel="Indice (Anno Base 2011 = 100)")

    # Plot dati reali
    for city in citta_di_riferimento
        rent_col = Symbol("indexed_$(lowercase(string(city)))_rent")
        Plots.plot!(p_rent, real_data_df.Year, real_data_df[!, rent_col], label=string(city), linestyle=:dash, lw=2)
        Plots.plot!(p_income, real_data_df.Year, real_data_df[!, rent_col], label="", linestyle=:dash, lw=2)
    end
    
    # Plot dei migliori N seed, usando i dati salvati
    sim_colors = [:red, :blue, :green, :purple, :orange]
    for (i, row) in enumerate(eachrow(first(sorted_results, top_n_to_plot)))
        anni_da_plottare_effettivi = 2011:(2011 + length(row.rent_series) - 1)
        color = sim_colors[mod1(i, length(sim_colors))]
        
        Plots.plot!(p_rent, anni_da_plottare_effettivi, row.rent_series, label="Seed $(row.seed)", lw=2.5, marker=:circle, markersize=3, color=color)
        Plots.plot!(p_income, anni_da_plottare_effettivi, row.income_series, label="", lw=2.5, marker=:circle, markersize=3, color=color)
    end

    final_plot = Plots.plot(p_rent, p_income, layout=(1,2), size=(1500, 600), legend=:outerright)
    display(final_plot)   


    results.combined_rmse = results.avg_rmse_rent .+ results.avg_rmse_income
    results.combined_shape = results.avg_shape_dist_rent .+ results.avg_shape_dist_income


    return results
end

function run_calibration_experiment(
    seeds_to_test::Vector{Int},
    real_data_df::DataFrame;
    # Parametri per la distribuzione empirica
    #income_bins::Vector,
    #income_weights::Vector,
    # Altri parametri di simulazione
    steps_per_year::Int = 8,
    n_steps_term::Int = 15
)
    println("--- Avvio Esperimento di Calibrazione su Larga Scala ---")
    
    # DataFrame per salvare i risultati finali di ogni seed
    results = DataFrame(
        seed=Int[], 
        avg_rmse_rent=Float64[], avg_rmse_income=Float64[],
        avg_shape_dist_rent=Float64[], avg_shape_dist_income=Float64[],
        rent_series=Vector[], income_series=Vector[],
        final_rent_index=Float64[], final_income_index=Float64[]
    )
    
    citta_di_riferimento = [:Brooklyn, :DC, :Houston, :Austin, :Philly]

    for seed in seeds_to_test
        print("  -> Esecuzione Seed: $seed...")
        
        # --- Esecuzione con la nuova initialize_model ---
        n_prerun_years = 4 
        n_interest_years = 10
        n_steps_total_dinamica = (n_prerun_years + n_interest_years) * steps_per_year

        model = initialize_model(
            #income_bin_edges_k = income_bins,
            #income_bin_weights = income_weights,
            #w_poor_percentile = w_poor_perc,
            #target_vacancy_rate = target_vac_rate,
            #base_potential_rent = base_rent,
            seed = seed
        )
        
        # Le fasi di termalizzazione e dinamica rimangono simili
        _, _, _, thermalized_model = run_simulation_phase_and_collect_data(model, n_steps_term, agent_step!, model_step_termalizzazione!, 0, "")
        _, model_metrics_df, _, _ = run_simulation_phase_and_collect_data(thermalized_model, n_steps_total_dinamica, agent_step!, model_step_dinamica_free!, n_steps_term, "")
        
        # --- Processamento e Calcolo Metriche (logica invariata) ---
        raw_rent = model_metrics_df.median_inner_city_rent
        raw_income = model_metrics_df.median_income_inner_city
        n_anni_effettivi = floor(Int, length(raw_rent) / steps_per_year)
        if n_anni_effettivi < 5; push!(results, [seed, Inf, Inf, Inf, Inf, [], [], NaN, NaN]); continue; end
        
        sim_annual_rent = [mean(raw_rent[((i-1)*steps_per_year+1):(i*steps_per_year)]) for i in 1:n_anni_effettivi]
        sim_annual_income = [mean(raw_income[((i-1)*steps_per_year+1):(i*steps_per_year)]) for i in 1:n_anni_effettivi]
        sim_5yr_rent_raw = rollmean(sim_annual_rent, 5)
        sim_5yr_income_raw = rollmean(sim_annual_income, 5)
        indexed_sim_rent = (sim_5yr_rent_raw ./ sim_5yr_rent_raw[1]) .* 100
        indexed_sim_income = (sim_5yr_income_raw ./ sim_5yr_income_raw[1]) .* 100
        final_rent = isempty(indexed_sim_rent) ? NaN : last(indexed_sim_rent)
        final_income = isempty(indexed_sim_income) ? NaN : last(indexed_sim_income)

        list_rmse_r, list_rmse_i, list_shape_r, list_shape_i = Float64[], Float64[], Float64[], Float64[]
        for city in citta_di_riferimento
            real_rent_series = real_data_df[!, Symbol("indexed_$(lowercase(string(city)))_rent")]
            real_income_series = real_data_df[!, Symbol("indexed_$(lowercase(string(city)))_income")]
            push!(list_rmse_r, calculate_rmse(indexed_sim_rent, real_rent_series))
            push!(list_rmse_i, calculate_rmse(indexed_sim_income, real_income_series))
            push!(list_shape_r, calculate_shape_distance(indexed_sim_rent, real_rent_series))
            push!(list_shape_i, calculate_shape_distance(indexed_sim_income, real_income_series))
        end
        
        push!(results, [seed, mean(list_rmse_r), mean(list_rmse_i), mean(list_shape_r), mean(list_shape_i), indexed_sim_rent, indexed_sim_income, final_rent, final_income])
        print(" Fatto.\n")
    end

    # Aggiungi le colonne combinate che servono per il clustering
    results.combined_rmse = results.avg_rmse_rent .+ results.avg_rmse_income
    results.combined_shape = results.avg_shape_dist_rent .+ results.avg_shape_dist_income

    return results
end


# Aggiunto `good_seeds` come terzo argomento
function cluster_simulation_outputs(results_df::DataFrame, k::Int, good_seeds::Vector{Int})
    println("\n--- Avvio Clustering dei Risultati della Simulazione in $k gruppi ---")

    # --- NESSUNA MODIFICA IN QUESTA SEZIONE (Passi 1, 2, 3) ---
    # 1. Seleziona le "features" per il clustering
    features_to_cluster = results_df[!, [:combined_rmse, :combined_shape, :final_rent_index, :final_income_index]]
    
    valid_rows_indices = completecases(features_to_cluster)
    features_clean = features_to_cluster[valid_rows_indices, :]
    original_indices = findall(valid_rows_indices)

    if nrow(features_clean) < k
        println("ERRORE: Meno dati validi ($nrow(features_clean)) del numero di cluster richiesti ($k).")
        return results_df
    end

    # 2. NORMALIZZAZIONE DEI DATI
    normalized_features = DataFrame()
    for col in names(features_clean)
        col_data = features_clean[!, col]
        min_val, max_val = minimum(col_data), maximum(col_data)
        if min_val ≈ max_val
            normalized_features[!, col] = fill(0.5, nrow(features_clean))
        else
            normalized_features[!, col] = (col_data .- min_val) ./ (max_val - min_val)
        end
    end

    # 3. Esegui il Clustering K-Means
    features_matrix = Matrix(normalized_features)'
    kmeans_result = kmeans(features_matrix, k; maxiter=200, display=:iter)
    assignments = kmeans_result.assignments

    # 4. Aggiungi i risultati al DataFrame originale
    annotated_df = copy(results_df)
    annotated_df.cluster .= 0
    for (i, original_idx) in enumerate(original_indices)
        annotated_df[original_idx, :cluster] = assignments[i]
    end

    # ========================== INIZIO MODIFICA ==========================
    # Abbiamo sostituito il vecchio blocco di riepilogo con la logica
    # della "Tabella Master", integrandola direttamente qui.

    println("\n--- Tabella di Riepilogo Dettagliata (Ordinata per Cluster) ---")

    # b. Aggiungi una colonna per identificare subito i "good seeds"
    annotated_df.is_good_seed = [row.seed in good_seeds for row in eachrow(annotated_df)]

    # c. Seleziona e riordina le colonne per la massima leggibilità
    colonne_da_mostrare = [
        :seed,
        :cluster,
        :is_good_seed,
        :combined_rmse,
        :combined_shape,
        :final_rent_index,
        :final_income_index
    ]

    # d. Ordina il DataFrame per cluster e poi per RMSE
    summary_ordinato = sort(annotated_df[!, colonne_da_mostrare], [:cluster, :combined_rmse])

    # e. Mostra la tabella finale e più leggibile
    display(summary_ordinato)
    
    # =========================== FINE MODIFICA ===========================

    return annotated_df
end

#= 1. Come Funziona il K-Means Clustering (Spiegato con un'Analogia)
Immagina di organizzare una festa per i tuoi seed. Ogni seed è un invitato che entra nella sala. Tu non li conosci, ma vuoi dividerli in k gruppi (diciamo 4) di persone che si "somigliano". Ogni invitato è descritto dalle sue 4 caratteristiche (RMSE, Shape Distance, Final Rent, Final Income).

L'algoritmo K-Means funziona così, passo per passo:

Passo 1: Scegli k e Posiziona i "Segnaposto"

Tu (l'organizzatore): Decidi che vuoi 4 gruppi. Quindi k=4.
K-Means: Sceglie a caso 4 invitati dalla folla e li nomina "capigruppo" temporanei. In termini tecnici, questi sono i centroidi iniziali. Immagina di mettere 4 tavoli a caso nella sala.
Passo 2: Prima Assegnazione (Assignment)

K-Means: Urla a tutti gli invitati: "Ognuno di voi vada al tavolo (centroide) a cui è più vicino!".
Come si misura la "vicinanza"? Con la distanza euclidea nello spazio a 4 dimensioni. Un invitato con caratteristiche [10, 5, 150, 200] sarà molto più "vicino" a un centroide [12, 6, 155, 205] che a uno [50, 40, 110, 120].
Alla fine di questo passo, ogni invitato è in un gruppo, anche se i gruppi sono ancora un po' disordinati.
Passo 3: Aggiornamento dei Centroidi (Update)

K-Means: Ora guarda ogni tavolo e dice: "Il centro di questo gruppo non è più dove avevo messo il tavolo all'inizio".
Per ogni gruppo, calcola il vero centro di gravità, ovvero la media di tutte le caratteristiche degli invitati in quel gruppo.
Sposta il tavolo (il centroide) in questa nuova posizione media. Il centroide ora rappresenta molto meglio il gruppo a cui è associato.
Passo 4: Ripetizione fino alla Convergenza

Ora che i tavoli si sono spostati, alcuni invitati ai bordi dei gruppi potrebbero scoprire di essere più vicini a un altro tavolo.
L'algoritmo ripete i Passi 2 e 3:
Ri-assegnazione: "Tutti controllino di nuovo qual è il tavolo più vicino! Se necessario, cambiate gruppo!". Alcuni invitati cambieranno tavolo.
Ri-aggiornamento: I centri dei tavoli vengono ricalcolati e spostati di nuovo.
Questo ciclo continua finché non si raggiunge una situazione stabile, ovvero un'intera iterazione in cui nessun invitato cambia più gruppo. A quel punto, i cluster sono definiti.
In sintesi, K-Means è un processo iterativo che affina progressivamente la posizione dei centri dei cluster e l'appartenenza dei punti ai cluster fino a trovare la configurazione più stabile e compatta possibile.=#


# Versione migliorata della funzione di plot con etichette
#=
function plot_cluster_pairs(df::DataFrame)
    features = [:combined_rmse, :combined_shape, :final_rent_index, :final_income_index]
    plot_list = []

    # Trasforma i seed in stringhe per le etichette
    labels = string.(df.seed)

    for i in 1:length(features)
        for j in (i+1):length(features)
            f1 = features[i]
            f2 = features[j]
            
            p = Plots.scatter(df[!, f1], df[!, f2],
                group = df.cluster,
                xlabel = string(f1),
                ylabel = string(f2),
                title = "$(f1) vs $(f2)",
                legend = :outertopright, 
                markersize = 5,
                # Aggiungiamo le etichette!
                series_annotations = Plots.text.(labels, :bottom, :left, 8)
            )
            push!(plot_list, p)
        end
    end

    if !isempty(plot_list)
        n_plots = length(plot_list)
        cols = ceil(Int, sqrt(n_plots))
        rows = ceil(Int, n_plots / cols)
        display(Plots.plot(plot_list..., layout=(rows, cols), size=(1600, 1000)))
    else
        println("Nessun cluster da plottare.")
    end
end=#


function plot_cluster_pairs(df::DataFrame)
    features = [:combined_rmse, :combined_shape, :final_rent_index, :final_income_index]
    plot_list = []

    # Trasforma i seed in stringhe per le etichette (se decidi di usarle)
    labels = string.(df.seed)

    # --- NUOVO: Definisci la palette di colori per coerenza ---
    cluster_colors = palette([:blue, :red, :green]) # Assumendo Cluster 1=blu, 2=rosso, 3=verde

    for i in 1:length(features)
        for j in (i+1):length(features)
            f1 = features[i]
            f2 = features[j]
            
            p = Plots.scatter(df[!, f1], df[!, f2],
                group = df.cluster,
                xlabel = string(f1),
                ylabel = string(f2),
                xguidefontsize = 20,   # dimensione label asse x
                yguidefontsize = 20,
                #title = "$(f1) vs $(f2)",
                legend = :outertopright, 
                markersize = 5,
                palette = cluster_colors # <-- APPLICA I COLORI PERSONALIZZATI
                # Per un grafico più pulito, le etichette dei seed sono disattivate.
                # Per riattivarle, rimuovi il commento dalla riga seguente.
                # series_annotations = Plots.text.(labels, :bottom, :left, 8)
            )
            push!(plot_list, p)
        end
    end

    if !isempty(plot_list)
        n_plots = length(plot_list)
        cols = 3 # Fissiamo a 3 colonne per un layout più pulito
        rows = ceil(Int, n_plots / cols)
        
        # --- MODIFICHE: Aggiungi i margini per dare più spazio ---
        display(Plots.plot(plot_list..., 
                           layout=(rows, cols), 
                           size=(1800, 1000),
                           left_margin=10Plots.mm,
                           bottom_margin=10Plots.mm
                           ))
    else
        println("Nessun cluster da plottare.")
    end
end



function plot_cluster_trends(clustered_df::DataFrame, real_df::DataFrame)
    println("--- Generating Cluster Comparison Plots (No Legend) ---")

    # Mappa di colori e stili di linea (invariata)
    cluster_colors = Dict(1 => :blue, 2 => :red, 3 => :green, 0 => :black)
    line_styles = [:solid, :dash, :dot, :dashdot]

    # --- MODIFICA: Rimuovi la legenda e traduci i titoli ---
    p_rent = Plots.plot(
        #title="Rent Trends by Cluster",
        xlabel="Year",
        ylabel="Rent Percentage Growth",
        legend=false,
        left_margin = 10Plots.mm,
        titlefontsize = 14,
        guidefontsize = 24,
        tickfontsize = 19,
        #legendfontsize = 15
    )
    p_income = Plots.plot(
        #title="Income Trends by Cluster",
        xlabel="Year",
        ylabel="Income Percentage Growth",
        legend=false,
        left_margin = 10Plots.mm,
        titlefontsize = 14,
        guidefontsize = 24,
        tickfontsize = 19
    )

    
    # Aggiunta dei trend dei cluster (rimossa l'etichetta)
    for group in groupby(clustered_df, :cluster)
        cluster_id = first(group).cluster
        colore_base = get(cluster_colors, cluster_id, :purple)

        for (i, row) in enumerate(eachrow(group))
            stile_linea = line_styles[mod1(i, length(line_styles))]
            anni_simulazione = 2011:(2011 + length(row.rent_series) - 1)
            
            Plots.plot!(p_rent, anni_simulazione, row.rent_series,
                        label="", # <-- Etichetta rimossa
                        color=colore_base, 
                        linestyle=stile_linea,
                        alpha=0.05,
                        lw=1.5)
            
            Plots.plot!(p_income, anni_simulazione, row.income_series,
                        label="", # <-- Etichetta rimossa
                        color=colore_base, 
                        linestyle=stile_linea,
                        alpha=0.1,
                        lw=1.5
                        )
        end
    end

    citta_reali = unique([split(c, "_")[2] for c in names(real_df, r"_rent")])
    for city in citta_reali
        rent_col = Symbol("indexed_$(city)_rent")
        income_col = Symbol("indexed_$(city)_income")
        Plots.plot!(p_rent, real_df.Year, real_df[!, rent_col], label="", #=color=:gray, linestyle=:dash, alpha=1=#)
        Plots.plot!(p_income, real_df.Year, real_df[!, income_col], label="", #=color=:gray, linestyle=:dash, alpha=1=#)
    end

    display(p_rent)
    display(p_income)
     #return p_rent, p_income
    # --- MODIFICA: Unione e visualizzazione con layout verticale ---
    #final_plot = Plots.plot(p_rent, p_income, layout=(2, 1), size=(900, 1200))
    
end

function plot_cluster_trends2(clustered_df::DataFrame, real_df::DataFrame, real_label_map::Dict)
    println("--- Generating Background Clusters with Real Data Overlay ---")

    cluster_colors = Dict(1 => :blue, 2 => :red, 3 => :green, 0 => :black)
    
    # 1. Inizializzazione Plot
    p_rent = Plots.plot(xlabel="Year", ylabel="Rent Percentage Growth", 
                        legend=:topleft, guidefontsize=16, tickfontsize=12, legendfontsize=10)
    p_income = Plots.plot(xlabel="Year", ylabel="Income Percentage Growth", 
                          legend=:topleft, guidefontsize=16, tickfontsize=12, legendfontsize=10)

    # 2. SFONDO: Disegna le serie delle simulazioni (Sfumate)
    # Disegniamo queste per prime così restano "sotto"
    for group in groupby(clustered_df, :cluster)
        cluster_id = first(group).cluster
        colore_base = get(cluster_colors, cluster_id, :gray)

        for row in eachrow(group)
            anni_simulazione = 2011:(2011 + length(row.rent_series) - 1)
            
            # Alpha molto basso (0.05) crea l'effetto area sfumata/nuvola
            Plots.plot!(p_rent, anni_simulazione, row.rent_series, 
                        label="", color=colore_base, alpha=0.08, lw=1)
            
            Plots.plot!(p_income, anni_simulazione, row.income_series, 
                        label="", color=colore_base, alpha=0.08, lw=1)
        end
    end

    #colors = [:red, :black, :green, :yellow, :orange]

    # 3. PRIMO PIANO: Dati Reali (Nitidi con Legenda)
    # Cicliamo sulle colonne del DataFrame dei dati 
    #color_counter = 1
    for col in names(real_df, r"indexed_.*_rent")
        # Estraiamo la chiave per la legenda (es. "brooklyn")
        key = replace(replace(string(col), "indexed_" => ""), "_rent" => "")
        label = get(real_label_map, key, key)
        
        # Stile: Dash per i "Control", Solid per gli altri
        style = contains(label, "Control") ? :dash : :solid
        
        # Disegniamo sopra le simulazioni (alpha=1.0 e lw=3 per farle risaltare)
        Plots.plot!(p_rent, real_df.Year, real_df[!, col], 
                    label=label, lw=3, linestyle=style, #=color = colors[color_counter],=# palette = :rainbow, alpha=1.0)
        #color_counter += 1
    end

    #color_counter = 1
    for col in names(real_df, r"indexed_.*_income")
        key = replace(replace(string(col), "indexed_" => ""), "_income" => "")
        label = get(real_label_map, key, key)
        style = contains(label, "Control") ? :dash : :solid
        
        Plots.plot!(p_income, real_df.Year, real_df[!, col], 
                    label=label, lw=3, linestyle=style, #=color = colors[color_counter],=#palette = :rainbow, alpha=1.0)
        #color_counter += 1
    end

    # Visualizzazione
    display(p_rent)
    display(p_income)
    
    return p_rent, p_income

    # Se vuoi salvarli per Overleaf come PDF:
    # savefig(p_rent, "rent_comparison.pdf")
    # savefig(p_income, "income_comparison.pdf")
end

function plot_cluster_trends3(clustered_df::DataFrame, real_df::DataFrame, real_label_map::Dict)
    println("--- Generating Shaded Cluster Envelopes with Real Data Overlay ---")

    cluster_colors = Dict(1 => :blue, 2 => :red, 3 => :green, 0 => :black)
    
    # 1. Inizializzazione Plot
    p_rent = Plots.plot(xlabel="Year", ylabel="Rent Percentage Growth", 
                        legend=:topleft, guidefontsize=16, tickfontsize=12, legendfontsize=10)
    p_income = Plots.plot(xlabel="Year", ylabel="Income Percentage Growth", 
                          legend=:topleft, guidefontsize=16, tickfontsize=12, legendfontsize=10)

    # 2. SFONDO: Aree dei Cluster (Ribbons)
    for group in groupby(clustered_df, :cluster)
        cluster_id = first(group).cluster
        colore_base = get(cluster_colors, cluster_id, :gray)

        # Calcolo dell'inviluppo per RENT
        # Trasformiamo la colonna di array in una matrice (Anni x Simulazioni)
        matrice_rent = hcat(group.rent_series...)
        min_rent = minimum(matrice_rent, dims=2)[:]
        max_rent = maximum(matrice_rent, dims=2)[:]
        
        # Calcolo dell'inviluppo per INCOME
        matrice_income = hcat(group.income_series...)
        min_income = minimum(matrice_income, dims=2)[:]
        max_income = maximum(matrice_income, dims=2)[:]

        anni_sim = 2011:(2011 + length(min_rent) - 1)

        # Disegniamo l'area sfumata (fillrange)
        Plots.plot!(p_rent, anni_sim, min_rent, fillrange=max_rent, 
                    fillalpha=0.25, color=colore_base, linealpha=0, label=false)
        
        Plots.plot!(p_income, anni_sim, min_income, fillrange=max_income, 
                    fillalpha=0.25, color=colore_base, linealpha=0, label=false)
    end

    # 3. PRIMO PIANO: Dati Reali (Palette Rainbow come richiesto)
    for col in names(real_df, r"indexed_.*_rent")
        key = replace(replace(string(col), "indexed_" => ""), "_rent" => "")
        label = get(real_label_map, key, key)
        style = contains(label, "Control") ? :dash : :solid
        
        Plots.plot!(p_rent, real_df.Year, real_df[!, col], 
                    label=label, lw=3, linestyle=style, palette=:rainbow, alpha=1.0)
    end

    for col in names(real_df, r"indexed_.*_income")
        key = replace(replace(string(col), "indexed_" => ""), "_income" => "")
        label = get(real_label_map, key, key)
        style = contains(label, "Control") ? :dash : :solid
        
        Plots.plot!(p_income, real_df.Year, real_df[!, col], 
                    label=label, lw=3, linestyle=style, palette=:rainbow, alpha=1.0)
    end

    display(p_rent)
    display(p_income)
    
    return p_rent, p_income
end

function get_initial_inner_city_amenity_score(model::ABM)
    
    total_amenity_score = 0.0
    num_houses = 0
    
    # Ricaviamo i confini dell'inner city (devono corrispondere a quelli in initialize_model)
    dims = model.grid_dims
    inner_r_start = floor(Int, dims[1] / 4) + 1
    inner_r_end   = floor(Int, dims[1] * 3 / 4)
    inner_c_start = floor(Int, dims[2] / 4) + 1
    inner_c_end   = floor(Int, dims[2] * 3 / 4)

    # Iteriamo su tutte le patch dell'inner city
    for r in inner_r_start:inner_r_end
        for c in inner_c_start:inner_c_end
            # Controlliamo che la patch sia una casa
            if model.patch_type[r, c] == :house
                total_amenity_score += model.raw_amenity_scores_matrix[r, c]
                num_houses += 1
            end
        end
    end
    
    # Calcoliamo la media
    if num_houses == 0
        return 0.0
    end
    
    return total_amenity_score / num_houses
end


function get_initial_public_amenity_value(model::ABM)
    
    total_public_amenity = 0.0
    num_public_spaces = 0
    
    # Ricaviamo i confini dell'inner city
    dims = model.grid_dims
    inner_r_start = floor(Int, dims[1] / 4) + 1
    inner_r_end   = floor(Int, dims[1] * 3 / 4)
    inner_c_start = floor(Int, dims[2] / 4) + 1
    inner_c_end   = floor(Int, dims[2] * 3 / 4)

    # Iteriamo su tutte le patch dell'inner city
    for r in inner_r_start:inner_r_end
        for c in inner_c_start:inner_c_end
            # Controlliamo che la patch sia uno spazio pubblico
            if model.patch_type[r, c] == :public
                # Usiamo `amenity_level` che contiene il valore intrinseco dello spazio
                total_public_amenity += model.amenity_level[r, c]
                num_public_spaces += 1
            end
        end
    end
    
    # Calcoliamo la media
    if num_public_spaces == 0
        return 0.0
    end
    
    return total_public_amenity / num_public_spaces
end


# -----------------------------------------------------------------------------
# FUNZIONE DI SUPPORTO
# Definiamo prima la funzione "helper" che fa il lavoro pesante.
# Il suo unico scopo è prendere una lista di posizioni e contare i cluster.
# -----------------------------------------------------------------------------
function count_clusters_from_positions(positions::Vector{NTuple{2, Int}}, model_ref::ABM)
    if isempty(positions)
        return 0
    end
    
    g = SimpleGraph(length(positions))
    pos_to_vertex = Dict(pos => i for (i, pos) in enumerate(positions))

    for (vertex_idx, pos) in enumerate(positions)
        for neighbor_pos in nearby_positions(pos, model_ref)
            if haskey(pos_to_vertex, neighbor_pos)
                neighbor_vertex_idx = pos_to_vertex[neighbor_pos]
                add_edge!(g, vertex_idx, neighbor_vertex_idx)
            end
        end
    end
    
    return length(connected_components(g))
end


# -----------------------------------------------------------------------------
# FUNZIONE PRINCIPALE
# Ora la funzione principale è più snella e facile da leggere.
# Il suo unico scopo è categorizzare le aree e chiamare la funzione di supporto.
# -----------------------------------------------------------------------------
function count_zonal_public_clusters(model::ABM)
    
    # 1. Suddividiamo le aree pubbliche in inner city e suburbs
    inner_city_spaces = NTuple{2, Int}[]
    suburb_spaces = NTuple{2, Int}[]
    
    # ... (tutta la logica per definire i confini e popolare le due liste rimane identica) ...
    dims = model.grid_dims
    cbd_center = model.cbd_center
    cbd_r = model.cbd_radius
    inner_r_start = floor(Int, dims[1] / 4) + 1
    inner_r_end   = floor(Int, dims[1] * 3 / 4)
    inner_c_start = floor(Int, dims[2] / 4) + 1
    inner_c_end   = floor(Int, dims[2] * 3 / 4)

    for r in 1:dims[1], c in 1:dims[2]
        if model.patch_type[r, c] == :public
            pos = (r, c)
            dist_from_center = sqrt((r - cbd_center[1])^2 + (c - cbd_center[2])^2)
            if dist_from_center <= cbd_r
                continue # Ignora le aree pubbliche nel CBD
            end

            if (inner_r_start <= r <= inner_r_end) && (inner_c_start <= c <= inner_c_end)
                push!(inner_city_spaces, pos)
            else
                push!(suburb_spaces, pos)
            end
        end
    end

    # 2. Chiamiamo la funzione di supporto per ogni zona
    n_clusters_inner = count_clusters_from_positions(inner_city_spaces, model)
    n_clusters_suburbs = count_clusters_from_positions(suburb_spaces, model)
    
    # 3. Restituiamo i risultati
    return (inner_city=n_clusters_inner, suburbs=n_clusters_suburbs)
end


function get_initial_population_counts(model::ABM)
    n_low_inner, n_high_inner = 0, 0
    n_low_suburbs, n_high_suburbs = 0, 0

    # Definiamo i confini
    dims = model.grid_dims
    cbd_center = model.cbd_center
    cbd_r = model.cbd_radius
    inner_r_start = floor(Int, dims[1] / 4) + 1
    inner_r_end   = floor(Int, dims[1] * 3 / 4)
    inner_c_start = floor(Int, dims[2] / 4) + 1
    inner_c_end   = floor(Int, dims[2] * 3 / 4)

    for agent in allagents(model)
        r, c = agent.pos
        # Controlliamo se la posizione è nell'inner city
        is_in_inner_city = (inner_r_start <= r <= inner_r_end) && (inner_c_start <= c <= inner_c_end)
        
        # Escludiamo il CBD
        dist_from_center = sqrt((r - cbd_center[1])^2 + (c - cbd_center[2])^2)
        if dist_from_center <= cbd_r
            continue
        end

        if is_in_inner_city
            if agent isa LowIncome
                n_low_inner += 1
            else
                n_high_inner += 1
            end
        else # Altrimenti è nei suburbs
            if agent isa LowIncome
                n_low_suburbs += 1
            else
                n_high_suburbs += 1
            end
        end
    end
    
    return (
        n_low_inner=n_low_inner, 
        n_high_inner=n_high_inner, 
        n_low_suburbs=n_low_suburbs, 
        n_high_suburbs=n_high_suburbs
    )
end


function get_initial_potential_rent_avg(model::ABM)
    total_potential_rent = 0.0
    num_houses = 0

    # Definiamo i confini dell'inner city
    dims = model.grid_dims
    inner_r_start = floor(Int, dims[1] / 4) + 1
    inner_r_end   = floor(Int, dims[1] * 3 / 4)
    inner_c_start = floor(Int, dims[2] / 4) + 1
    inner_c_end   = floor(Int, dims[2] * 3 / 4)

    for r in inner_r_start:inner_r_end
        for c in inner_c_start:inner_c_end
            if model.patch_type[r, c] == :house
                # Usiamo la matrice potential_rent del modello
                total_potential_rent += model.potential_rent[r, c]
                num_houses += 1
            end
        end
    end

    return num_houses > 0 ? total_potential_rent / num_houses : 0.0
end

function get_initial_house_age_avg(model::ABM)
    total_age = 0.0
    num_houses = 0

    # Definiamo i confini dell'inner city
    dims = model.grid_dims
    inner_r_start = floor(Int, dims[1] / 4) + 1
    inner_r_end   = floor(Int, dims[1] * 3 / 4)
    inner_c_start = floor(Int, dims[2] / 4) + 1
    inner_c_end   = floor(Int, dims[2] * 3 / 4)

    for r in inner_r_start:inner_r_end
        for c in inner_c_start:inner_c_end
            if model.patch_type[r, c] == :house
                # Usiamo la matrice house_age del modello
                total_age += model.house_age[r, c]
                num_houses += 1
            end
        end
    end

    return num_houses > 0 ? total_age / num_houses : 0.0
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

function plot_quadrant_segregation_by_amenity(
    summary_df::DataFrame;
    step_to_plot::Int,
    grid_dims::NTuple{2, Int}, # Usiamo questo per calcolare le dimensioni del quadrante
    title_suffix::String = ""
)
    if isempty(summary_df)
        println("DataFrame riassuntivo per la segregazione per quadrante è vuoto. Impossibile generare il plot.")
        return nothing
    end

    # Filtra il DataFrame per lo step desiderato
    df_to_plot = filter(row -> row.step == step_to_plot, summary_df)

    if isempty(df_to_plot)
        println("Nessun dato trovato per lo step $step_to_plot nel DataFrame riassuntivo per i quadranti.")
        return nothing
    end

    # Calcola le proporzioni (rispetto al totale degli agenti nel quadrante)
    df_to_plot.total_agents_in_quadrant = df_to_plot.n_low_quadrant + df_to_plot.n_high_quadrant
    
    df_to_plot.prop_low_in_quadrant = [
        t_q > 0 ? n_l_q / t_q : 0.0
        for (n_l_q, t_q) in zip(df_to_plot.n_low_quadrant, df_to_plot.total_agents_in_quadrant)
    ]
    df_to_plot.prop_high_in_quadrant = [
        t_q > 0 ? n_h_q / t_q : 0.0
        for (n_h_q, t_q) in zip(df_to_plot.n_high_quadrant, df_to_plot.total_agents_in_quadrant)
    ]

    # --- INIZIO: Calcolo automatico dei limiti dell'asse X per l'amenity media ---
    all_amenity_data = skipnan(df_to_plot.mean_quadrant_amenity)
    
    x_lower_bound = 0.0
    x_upper_bound = 0.0

    if !isempty(all_amenity_data)
        min_amenity = minimum(all_amenity_data)
        max_amenity = maximum(all_amenity_data)

        x_lower_bound = max(0.0, min_amenity - (max_amenity - min_amenity) * 0.1)
        x_upper_bound = max(max_amenity + (max_amenity - min_amenity) * 0.1, x_lower_bound + 0.01)

        if (x_upper_bound - x_lower_bound) < 0.001 && x_lower_bound == 0.0
             x_upper_bound = 0.005
        elseif (x_upper_bound - x_lower_bound) < 0.001
             x_lower_bound = max(0.0, x_lower_bound - 0.001)
             x_upper_bound = x_upper_bound + 0.001
        end

    else
        x_upper_bound = 1.0
    end
    
    x_axis_limits_amenity = (x_lower_bound, x_upper_bound)
    println("Limiti X calcolati per amenity quadrante: $x_axis_limits_amenity")
    # --- FINE: Calcolo automatico dei limiti dell'asse X ---

    # --- Plot per Agenti Low Income ---
    plt_low_income = Plots.scatter(
        df_to_plot.mean_quadrant_amenity,
        df_to_plot.prop_low_in_quadrant,
        xlabel = "Amenity Media del Quadrante",
        ylabel = "Proporzione di Agenti Low Income nel Quadrante",
        title = "Distribuzione Low Income (Step: $step_to_plot)" * title_suffix,
        legend = false,
        marker = (7, :circle, :red, 0.7, Plots.stroke(0.5, :black)),
        size=(800, 600),
        xlims=x_axis_limits_amenity,
        ylims=(0, 1.05),
        grid=true,
        xticks=collect(0:0.1:1.0),
        yticks=collect(0:0.1:1.0)
    )

    for r in eachrow(df_to_plot)
        Plots.annotate!(r.mean_quadrant_amenity + (x_axis_limits_amenity[2] - x_axis_limits_amenity[1]) * 0.02,
                       r.prop_low_in_quadrant + 0.01,
                       Plots.text("($(r.quadrant_coords_x),$(r.quadrant_coords_y))", 6, :left, :black)) # <-- MODIFICATO: Plots.text
    end

    display(plt_low_income)

    # --- Plot per Agenti High Income ---
    plt_high_income = Plots.scatter(
        df_to_plot.mean_quadrant_amenity,
        df_to_plot.prop_high_in_quadrant,
        xlabel = "Amenity Media del Quadrante",
        ylabel = "Proporzione di Agenti High Income nel Quadrante",
        title = "Distribuzione High Income (Step: $step_to_plot)" * title_suffix,
        legend = false,
        marker = (7, :circle, :blue, 0.7, Plots.stroke(0.5, :black)),
        size=(800, 600),
        xlims=x_axis_limits_amenity,
        ylims=(0, 1.05),
        grid=true,
        xticks=collect(0:0.1:1.0),
        yticks=collect(0:0.1:1.0)
    )

    for r in eachrow(df_to_plot)
        Plots.annotate!(r.mean_quadrant_amenity + (x_axis_limits_amenity[2] - x_axis_limits_amenity[1]) * 0.02,
                       r.prop_high_in_quadrant + 0.01,
                       Plots.text("($(r.quadrant_coords_x),$(r.quadrant_coords_y))", 6, :left, :black)) # <-- MODIFICATO: Plots.text
    end

    display(plt_high_income)

    return plt_low_income, plt_high_income
end


function plot_multi_seed_quadrant_segregation(
    summary_quadrant_df::DataFrame;
    step_to_plot::Int,
    title_suffix::String = ""
)
    if isempty(summary_quadrant_df)
        println("DataFrame riassuntivo per la segregazione per quadrante multi-seed è vuoto. Impossibile generare il plot.")
        return nothing
    end

    # Filtra il DataFrame per lo step desiderato
    df_at_step = filter(row -> row.step == step_to_plot, summary_quadrant_df)

    if isempty(df_at_step)
        println("Nessun dato trovato per lo step $step_to_plot nel DataFrame riassuntivo per i quadranti.")
        return nothing
    end

    # Calcola le proporzioni medie di Low/High Income all'interno di ogni quadrante
    # (le colonne mean_n_low_quadrant e mean_n_high_quadrant sono già medie sui seed)
    df_at_step.mean_total_agents_in_quadrant = df_at_step.mean_n_low_quadrant + df_at_step.mean_n_high_quadrant
    
    df_at_step.mean_prop_low_in_quadrant = [
        t_q > 0 ? n_l_q / t_q : 0.0
        for (n_l_q, t_q) in zip(df_at_step.mean_n_low_quadrant, df_at_step.mean_total_agents_in_quadrant)
    ]
    df_at_step.mean_prop_high_in_quadrant = [
        t_q > 0 ? n_h_q / t_q : 0.0
        for (n_h_q, t_q) in zip(df_at_step.mean_n_high_quadrant, df_at_step.mean_total_agents_in_quadrant)
    ]

    # --- Calcolo automatico dei limiti dell'asse X (Amenity Media) ---
    all_amenity_data = skipnan(df_at_step.mean_quadrant_amenity)
    x_lower_bound = 0.0
    x_upper_bound = 0.0

    if !isempty(all_amenity_data)
        min_amenity = minimum(all_amenity_data)
        max_amenity = maximum(all_amenity_data)
        x_range = max_amenity - min_amenity
        
        x_lower_bound = max(0.0, min_amenity - x_range * 0.15) # Margine maggiore
        x_upper_bound = max(max_amenity + x_range * 0.15, x_lower_bound + 0.01)

        if (x_upper_bound - x_lower_bound) < 0.001 && x_lower_bound == 0.0
             x_upper_bound = 0.005
        elseif (x_upper_bound - x_lower_bound) < 0.001
             x_lower_bound = max(0.0, x_lower_bound - 0.001)
             x_upper_bound = x_upper_bound + 0.001
        end
    else
        x_upper_bound = 1.0
    end
    x_axis_limits_amenity = (x_lower_bound, x_upper_bound)
    println("Limiti X calcolati per amenity quadrante (Multi-Seed): $x_axis_limits_amenity")

    # --- Plot Combinato LowIncome e HighIncome ---
    plt = Plots.scatter(
        df_at_step.mean_quadrant_amenity,
        df_at_step.mean_prop_low_in_quadrant,
        label = "Low Income (Media)",
        marker = (7, :circle, :red, 0.7, Plots.stroke(0.5, :black)),
        xlabel = "Amenity Media del Quadrante",
        ylabel = "Proporzione di Agenti nel Quadrante",
        title = "Distribuzione Agenti per Amenity Quadrante (Step: $step_to_plot)" * title_suffix,
        size=(1000, 700),
        xlims=x_axis_limits_amenity,
        ylims=(0, 1.05),
        grid=true,
        xticks=collect(range(x_axis_limits_amenity[1], stop=x_axis_limits_amenity[2], length=5)), # Tick dinamici per X
        yticks=collect(0:0.1:1.0),
        legend=:topright
    )

    Plots.scatter!(plt,
        df_at_step.mean_quadrant_amenity,
        df_at_step.mean_prop_high_in_quadrant,
        label = "High Income (Media)",
        marker = (7, :square, :blue, 0.7, Plots.stroke(0.5, :black))
    )

    # --- Aggiungi linee di regressione lineare ---
    # Modello lineare: y = a + b*x
    linear_model(x, p) = p[1] .+ p[2] .* x

    # Regressione per Low Income
    low_data_x = skipnan(df_at_step.mean_quadrant_amenity)
    low_data_y = skipnan(df_at_step.mean_prop_low_in_quadrant)
    if length(low_data_x) > 1 && length(low_data_y) == length(low_data_x)
        fit_low = LsqFit.curve_fit(linear_model, low_data_x, low_data_y, [0.5, 0.0]) # [initial a, initial b]
        Plots.plot!(plt, low_data_x, linear_model(low_data_x, fit_low.param),
                    line=(:dash, 2, :red), label="Regr. Low Income")
    end

    # Regressione per High Income
    high_data_x = skipnan(df_at_step.mean_quadrant_amenity)
    high_data_y = skipnan(df_at_step.mean_prop_high_in_quadrant)
    if length(high_data_x) > 1 && length(high_data_y) == length(high_data_x)
        fit_high = LsqFit.curve_fit(linear_model, high_data_x, high_data_y, [0.5, 0.0])
        Plots.plot!(plt, high_data_x, linear_model(high_data_x, fit_high.param),
                    line=(:dash, 2, :blue), label="Regr. High Income")
    end

    display(plt)
    return plt
end

#=
function plot_raw_quadrant_segregation_with_regression(
    all_runs_quadrant_df::DataFrame;
    step_to_plot::Int,
    title_suffix::String = "",
    add_regression_line::Bool = true # NUOVO PARAMETRO
)
    if isempty(all_runs_quadrant_df)
        println("DataFrame 'all_runs_quadrant_metrics_df' è vuoto. Impossibile generare i plot di segregazione raw.")
        return nothing
    end

    df_at_step_raw = filter(row -> row.step == step_to_plot, all_runs_quadrant_df)

    if isempty(df_at_step_raw)
        println("Nessun dato trovato per lo step $step_to_plot nel DataFrame raw per i quadranti.")
        return nothing
    end

    df_at_step_raw.total_agents_in_quadrant = df_at_step_raw.n_low_quadrant + df_at_step_raw.n_high_quadrant
    
    df_at_step_raw.prop_low_in_quadrant = [
        t_q > 0 ? n_l_q / t_q : 0.0
        for (n_l_q, t_q) in zip(df_at_step_raw.n_low_quadrant, df_at_step_raw.total_agents_in_quadrant)
    ]
    df_at_step_raw.prop_high_in_quadrant = [
        t_q > 0 ? n_h_q / t_q : 0.0
        for (n_h_q, t_q) in zip(df_at_step_raw.n_high_quadrant, df_at_step_raw.total_agents_in_quadrant)
    ]

    all_amenity_data = skipnan(df_at_step_raw.mean_quadrant_amenity)
    x_lower_bound = 0.0
    x_upper_bound = 0.0

    if !isempty(all_amenity_data)
        min_amenity = minimum(all_amenity_data)
        max_amenity = maximum(all_amenity_data)
        x_range = max_amenity - min_amenity
        
        x_lower_bound = max(0.0, min_amenity - x_range * 0.15)
        x_upper_bound = max(max_amenity + x_range * 0.15, x_lower_bound + 0.01)

        if (x_upper_bound - x_lower_bound) < 0.001 && x_lower_bound == 0.0
             x_upper_bound = 0.005
        elseif (x_upper_bound - x_lower_bound) < 0.001
             x_lower_bound = max(0.0, x_lower_bound - 0.001)
             x_upper_bound = x_upper_bound + 0.001
        end
    else
        x_upper_bound = 1.0
    end
    x_axis_limits_amenity = (x_lower_bound, x_upper_bound)
    println("Limiti X calcolati per amenity quadrante (Raw): $x_axis_limits_amenity")

    # Modello lineare per regressione: y = a + b*x
    linear_model(x, p) = p[1] .+ p[2] .* x

    # --- Plot per Agenti Low Income (RAW) ---
    plt_low_income_raw = Plots.scatter(
        df_at_step_raw.mean_quadrant_amenity,
        df_at_step_raw.prop_low_in_quadrant,
        label = "Punti Individuali (Raw)",
        marker = (3, :circle, :red, 0.3),
        xlabel = "Amenity Media del Quadrante",
        ylabel = "Proporzione di Agenti Low Income nel Quadrante",
        title = "Distribuzione Raw Low Income (Step: $step_to_plot)" * title_suffix,
        size=(1000, 700),
        xlims=x_axis_limits_amenity,
        ylims=(0, 1.05),
        grid=true,
        xticks=collect(range(x_axis_limits_amenity[1], stop=x_axis_limits_amenity[2], length=5)),
        yticks=collect(0:0.1:1.0),
        legend=:topright
    )

    if add_regression_line # Condizionale per aggiungere la linea di regressione
        low_data_x_raw = skipnan(df_at_step_raw.mean_quadrant_amenity)
        low_data_y_raw = skipnan(df_at_step_raw.prop_low_in_quadrant)
        if length(low_data_x_raw) > 1 && length(low_data_y_raw) == length(low_data_x_raw)
            fit_low_raw = LsqFit.curve_fit(linear_model, low_data_x_raw, low_data_y_raw, [0.5, 0.0])
            Plots.plot!(plt_low_income_raw, low_data_x_raw, linear_model(low_data_x_raw, fit_low_raw.param),
                        line=(:solid, 3, :darkred), label="Regr. Lineare Low Income")
        end
    end

    display(plt_low_income_raw)

    # --- Plot per Agenti High Income (RAW) ---
    plt_high_income_raw = Plots.scatter(
        df_at_step_raw.mean_quadrant_amenity,
        df_at_step_raw.prop_high_in_quadrant,
        label = "Punti Individuali (Raw)",
        marker = (3, :square, :blue, 0.3),
        xlabel = "Amenity Media del Quadrante",
        ylabel = "Proporzione di Agenti High Income nel Quadrante",
        title = "Distribuzione Raw High Income (Step: $step_to_plot)" * title_suffix,
        size=(1000, 700),
        xlims=x_axis_limits_amenity,
        ylims=(0, 1.05),
        grid=true,
        xticks=collect(range(x_axis_limits_amenity[1], stop=x_axis_limits_amenity[2], length=5)),
        yticks=collect(0:0.1:1.0),
        legend=:topright
    )

    if add_regression_line # Condizionale per aggiungere la linea di regressione
        high_data_x_raw = skipnan(df_at_step_raw.mean_quadrant_amenity)
        high_data_y_raw = skipnan(df_at_step_raw.prop_high_in_quadrant)
        if length(high_data_x_raw) > 1 && length(high_data_y_raw) == length(high_data_x_raw)
            fit_high_raw = LsqFit.curve_fit(linear_model, high_data_x_raw, high_data_y_raw, [0.5, 0.0])
            Plots.plot!(plt_high_income_raw, high_data_x_raw, linear_model(high_data_x_raw, fit_high_raw.param),
                        line=(:solid, 3, :darkblue), label="Regr. Lineare High Income")
        end
    end

    display(plt_high_income_raw)

    return plt_low_income_raw, plt_high_income_raw
end=#

# Aggiungi `using StatsBase` all'inizio del file se non c'è già

function plot_raw_quadrant_segregation_with_regression(
    all_runs_quadrant_df::DataFrame;
    step_to_plot::Int,
    title_suffix::String = "",
    add_regression_line::Bool = true
)
    # ... (tutta la parte iniziale della funzione fino al calcolo del fit rimane invariata) ...
    # ... (df_at_step_raw, prop_low_in_quadrant, etc. vengono calcolati come prima) ...
    
    # (Assumiamo che il codice fino a qui sia lo stesso che mi hai mostrato)
    df_at_step_raw = filter(row -> row.step == step_to_plot, all_runs_quadrant_df)
    if isempty(df_at_step_raw); println("Nessun dato per lo step $step_to_plot."); return; end
    df_at_step_raw.total_agents_in_quadrant = df_at_step_raw.n_low_quadrant .+ df_at_step_raw.n_high_quadrant
    df_at_step_raw.prop_low_in_quadrant = [t > 0 ? n / t : 0.0 for (n, t) in zip(df_at_step_raw.n_low_quadrant, df_at_step_raw.total_agents_in_quadrant)]
    df_at_step_raw.prop_high_in_quadrant = [t > 0 ? n / t : 0.0 for (n, t) in zip(df_at_step_raw.n_high_quadrant, df_at_step_raw.total_agents_in_quadrant)]
    
    linear_model(x, p) = p[1] .+ p[2] .* x
    
    # --- Plot per Agenti High Income (con R²) ---
    
    high_data_x_raw = df_at_step_raw.mean_quadrant_amenity
    high_data_y_raw = df_at_step_raw.prop_high_in_quadrant
    
    r2_high_text = "" # Inizializza la stringa per R²
    if add_regression_line && length(high_data_x_raw) > 1
        fit_high_raw = LsqFit.curve_fit(linear_model, high_data_x_raw, high_data_y_raw, [0.5, 0.0])
        r2_val = calculate_r_squared(fit_high_raw, high_data_y_raw)
        r2_high_text = " (R² = $(round(r2_val, digits=2)))" # Formatta la stringa
    end

    plt_high_income_raw = Plots.scatter(
        high_data_x_raw, high_data_y_raw,
        label = "Punti Individuali (Raw)",
        marker = (3, :square, :blue, 0.3),
        xlabel = "Amenity Media del Quadrante",
        ylabel = "Proporzione Agenti High Income",
        # MODIFICA: Aggiungi R² al titolo
        title = "Distribuzione Raw High Income (Step: $step_to_plot)$(r2_high_text)" * title_suffix,
        # ... (il resto degli attributi del plot rimane invariato) ...
        size=(1000, 700), legend=:topleft
    )

    if add_regression_line && length(high_data_x_raw) > 1
        fit_high_raw = LsqFit.curve_fit(linear_model, high_data_x_raw, high_data_y_raw, [0.5, 0.0])
        Plots.plot!(plt_high_income_raw, high_data_x_raw, linear_model(high_data_x_raw, fit_high_raw.param),
                    line=(:solid, 3, :darkblue), label="Regr. Lineare")
    end
    
    display(plt_high_income_raw)
    
    # Puoi fare lo stesso per il plot dei Low Income
    # ... (codice per il plot dei Low Income con R²) ...

    return # La funzione può restituire i plot se necessario
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


#=
function run_multi_seed_scenario(
    seeds_to_run::Vector{Int},
    model_step_func::Function; # <- Qui passi la funzione di scenario!
    # --- Parametri di simulazione ---
    n_steps_term::Int = 15,
    n_steps_dyn::Int = 112,
    grid_dims::Tuple{Int, Int} = (40, 40),
    num_agents::Int = 800
)
    println("--- Avvio esecuzione multi-seed per lo scenario: $(string(model_step_func)) ---")
    
    # DataFrame per accumulare i risultati di tutti i seed
    all_runs_metrics_df = DataFrame()

    # Parametri comuni che non cambiano tra i seed
    w_poor_param = probs(create_mixed_distribution().prior)[1]
    income_dist = create_mixed_distribution()

    # --- Ciclo di Esecuzione ---
    for seed_val in seeds_to_run
        println("  -> Esecuzione simulazione per il seed: $seed_val")
        
        # Inizializza il modello per il seed corrente
        local model_instance = initialize_model(
            dims = grid_dims,
            N_agents = num_agents,
            income_distribution = income_dist,
            w_poor = w_poor_param,
            seed = seed_val
        )
        
        # Esegui termalizzazione e dinamica usando le funzioni helper che già hai
        _, model_metrics_term, _, thermalized_model = run_simulation_phase_and_collect_data(model_instance, n_steps_term, agent_step!, model_step_termalizzazione!, 0, "Term")
        
        # QUI LA MAGIA: usiamo la funzione passata come argomento per la fase dinamica
        _, model_metrics_dyn, _, _ = run_simulation_phase_and_collect_data(thermalized_model, n_steps_dyn, agent_step!, model_step_func, n_steps_term, "Dyn")
        
        # Combina i risultati della singola run e aggiungi la colonna del seed
        combined_df_for_seed = vcat(model_metrics_term, model_metrics_dyn[2:end, :])
        combined_df_for_seed.seed .= seed_val
        
        # Aggiungi i risultati di questa run al DataFrame principale
        append!(all_runs_metrics_df, combined_df_for_seed)
    end

    println("\n--- Esecuzione multi-seed completata per lo scenario: $(string(model_step_func)) ---")
    
    return all_runs_metrics_df
end=#

#=
function run_multi_seed_scenario(
    seeds_to_run::Vector{Int},
    model_step_func::Function;
    # --- Parametri di simulazione ---
    n_steps_term::Int = 15,
    n_steps_dyn::Int = 112,
    grid_dims::Tuple{Int, Int} = (40, 40),
    num_agents::Int = 800
)
    println("--- Avvio esecuzione multi-seed per lo scenario: $(string(model_step_func)) ---")
    
    # MODIFICA: Inizializziamo due DataFrame per raccogliere tutti i dati
    all_runs_model_metrics_df = DataFrame()
    all_runs_quadrant_df = DataFrame() # <-- NUOVO

    w_poor_param = probs(create_mixed_distribution().prior)[1]
    income_dist = create_mixed_distribution()

    for seed_val in seeds_to_run
        println("  -> Esecuzione simulazione per il seed: $seed_val")
        
        local model_instance = initialize_model(
            dims = grid_dims, N_agents = num_agents, income_distribution = income_dist,
            w_poor = w_poor_param, seed = seed_val
        )
        
        # Esegui le fasi e cattura tutti i DataFrame restituiti
        _, model_metrics_term, quadrant_df_term, thermalized_model = run_simulation_phase_and_collect_data(model_instance, n_steps_term, agent_step!, model_step_termalizzazione!, 0, "Term")
        _, model_metrics_dyn, quadrant_df_dyn, _ = run_simulation_phase_and_collect_data(thermalized_model, n_steps_dyn, agent_step!, model_step_func, n_steps_term, "Dyn")
        
        # Combina i dati per il MODELLO e aggiungi il seed
        combined_model_df = vcat(model_metrics_term, model_metrics_dyn[2:end, :])
        combined_model_df.seed .= seed_val
        append!(all_runs_model_metrics_df, combined_model_df)

        # MODIFICA: Combina i dati per i QUADRANTI e aggiungi il seed
        combined_quadrant_df = vcat(quadrant_df_term, quadrant_df_dyn[2:end, :])
        combined_quadrant_df.seed .= seed_val
        append!(all_runs_quadrant_df, combined_quadrant_df) # <-- NUOVO
    end

    println("\n--- Esecuzione multi-seed completata per: $(string(model_step_func)) ---")
    
    # MODIFICA: Restituisci entrambi i DataFrame
    return all_runs_model_metrics_df, all_runs_quadrant_df
end=#

function run_multi_seed_scenario(
    seeds_to_run::Vector{Int},
    model_step_func::Function;
    model_params::NamedTuple = (;), # Accetta parametri extra per initialize_model
    n_steps_term::Int = 15,
    n_steps_dyn::Int = 250, # Aumentiamo il default per questo esperimento
    grid_dims::Tuple{Int, Int} = (40, 40),
    num_agents::Int = 800
)
    println("--- Esecuzione per: $(string(model_step_func)) | Parametri: $(model_params) ---")
    
    all_runs_model_metrics_df = DataFrame()
    all_runs_quadrant_df = DataFrame()
    w_poor_param = probs(create_mixed_distribution().prior)[1]
    income_dist = create_mixed_distribution()

    for seed_val in seeds_to_run
        print("  -> Seed: $seed_val...") # Usiamo print per vedere l'avanzamento sulla stessa riga
        
        final_params = merge(
            (dims=grid_dims, N_agents=num_agents, income_distribution=income_dist, w_poor=w_poor_param, seed=seed_val),
            model_params
        )
        local model_instance = initialize_model(; final_params...)
        
        _, model_metrics_term, quadrant_df_term, thermalized_model = run_simulation_phase_and_collect_data(model_instance, n_steps_term, agent_step!, model_step_termalizzazione!, 0, "Term")
        _, model_metrics_dyn, quadrant_df_dyn, _ = run_simulation_phase_and_collect_data(thermalized_model, n_steps_dyn, agent_step!, model_step_func, n_steps_term, "Dyn")
        
        combined_model_df = vcat(model_metrics_term, model_metrics_dyn[2:end, :]); combined_model_df.seed .= seed_val
        append!(all_runs_model_metrics_df, combined_model_df)

        combined_quadrant_df = vcat(quadrant_df_term, quadrant_df_dyn[2:end, :]); combined_quadrant_df.seed .= seed_val
        append!(all_runs_quadrant_df, combined_quadrant_df)
        print(" Fatto.\n")
    end
    
    return all_runs_model_metrics_df, all_runs_quadrant_df
end

function run_multi_seed_scenario_new(
    seeds_to_run::Vector{Int},
    model_step_func::Function;
    
    # Parametri di esecuzione (invariati)
    n_steps_term::Int = 15,
    n_steps_dyn::Int = 112,
    public_housing_fraction = 0.15,
    public_housing_negative_spillover_boost = 2.0

)
    println("--- Avvio esecuzione per: $(string(model_step_func)) ---")
    
    all_runs_model_metrics_df = DataFrame()
    all_runs_quadrant_df = DataFrame()

    for seed_val in seeds_to_run
        print("  -> Seed: $seed_val...")
        
        # --- MODIFICA CHIAVE: Chiama `initialize_model` con i parametri corretti ---
        model_instance = initialize_model(
            seed = seed_val, 
            public_housing_fraction = public_housing_fraction,
            public_housing_negative_spillover_boost = public_housing_negative_spillover_boost
            # Nota: Non passiamo più N_agents, income_distribution, o w_poor
        )
        
        # Il resto della logica rimane identico
        _, model_metrics_term, quadrant_df_term, thermalized_model = run_simulation_phase_and_collect_data(model_instance, n_steps_term, agent_step!, model_step_termalizzazione!, 0, "Term")
        _, model_metrics_dyn, quadrant_df_dyn, _ = run_simulation_phase_and_collect_data(thermalized_model, n_steps_dyn, agent_step!, model_step_func, n_steps_term, "Dyn")
        
        combined_model_df = vcat(model_metrics_term, model_metrics_dyn[2:end, :]); combined_model_df.seed .= seed_val
        append!(all_runs_model_metrics_df, combined_model_df)

        combined_quadrant_df = vcat(quadrant_df_term, quadrant_df_dyn[2:end, :]); combined_quadrant_df.seed .= seed_val
        append!(all_runs_quadrant_df, combined_quadrant_df)
        
        print(" Fatto.\n")
    end
    
    return all_runs_model_metrics_df, all_runs_quadrant_df
end


function plot_best_seeds_verification(
    simulation_runs_df::DataFrame, 
    real_data_df::DataFrame,
    scenario_title::String
)
    println("--- Generazione grafico di verifica per lo scenario: $scenario_title ---")

    # --- 1. Processamento Dati Simulati (Logica Definitiva) ---
    processed_runs_df = DataFrame()
    steps_per_year = 8
    n_anni_da_mappare = 10 # Vogliamo 10 punti finali per il 2011-2020
    window_size_in_steps = 5 * steps_per_year # La finestra mobile è di 40 step

    for group in groupby(simulation_runs_df, :seed)
        df_seed = DataFrame(group)
        
        # Isola i dati della fase dinamica (dopo lo step 15)
        raw_rent_dyn = df_seed[df_seed.step .> 15, :median_inner_city_rent]
        raw_income_dyn = df_seed[df_seed.step .> 15, :median_income_inner_city]
        
        # Assicurati di avere abbastanza dati per il processo
        if length(raw_rent_dyn) < window_size_in_steps + (n_anni_da_mappare - 1) * steps_per_year
            println("ATTENZIONE: Dati insufficienti per il Seed $(first(df_seed.seed)). Saltato.")
            continue
        end

        # Calcola la media mobile sull'intera serie di dati grezzi
        rent_rolling_means = rolling(mean, raw_rent_dyn, window_size_in_steps)
        income_rolling_means = rolling(mean, raw_income_dyn, window_size_in_steps)

        # Estrai i valori della media mobile alla fine di ogni "anno"
        # Il 1° punto (2011) usa gli step 1-40, quindi è il 1° valore del rolling mean
        # Il 2° punto (2012) usa gli step 9-48, quindi è il 9° valore del rolling mean
        # e così via...
        indices_to_sample = [1 + (i - 1) * steps_per_year for i in 1:n_anni_da_mappare]
        
        sampled_rent_ma = rent_rolling_means[indices_to_sample]
        sampled_income_ma = income_rolling_means[indices_to_sample]

        # Indicizza i 10 punti ottenuti partendo da 100
        indexed_sim_rent = (sampled_rent_ma ./ sampled_rent_ma[1]) .* 100
        indexed_sim_income = (sampled_income_ma ./ sampled_income_ma[1]) .* 100
        
        # L'asse X corrisponde esattamente agli anni dei dati reali
        anni_reali_da_confrontare = minimum(real_data_df.Year):(minimum(real_data_df.Year) + n_anni_da_mappare - 1)

        temp_df = DataFrame(
            seed = first(df_seed.seed),
            Year = anni_reali_da_confrontare,
            indexed_rent = indexed_sim_rent,
            indexed_income = indexed_sim_income
        )
        append!(processed_runs_df, temp_df)
    end

    if isempty(processed_runs_df)
        println("ATTENZIONE: Nessun dato valido da plottare per lo scenario '$scenario_title'.")
        return
    end

    # Il resto della funzione di plotting è identico
    avg_trend_df = combine(groupby(processed_runs_df, :Year), :indexed_rent => mean => :mean_rent, :indexed_income => mean => :mean_income)
    
    anni_reali = real_data_df.Year
    p_rent = Plots.plot(title="Affitti: $scenario_title vs Dati Reali", xlabel="Anno", ylabel="Indice (Anno Base = 100)")
    p_income = Plots.plot(title="Redditi: $scenario_title vs Dati Reali", xlabel="Anno", ylabel="Indice (Anno Base = 100)")

    for city_col in names(real_data_df, r"_rent")
        Plots.plot!(p_rent, anni_reali, real_data_df[!, city_col], label=replace(string(city_col), r"indexed_|_rent" => ""), linestyle=:dash, lw=1.5)
    end
    for group in groupby(processed_runs_df, :seed)
        Plots.plot!(p_rent, group.Year, group.indexed_rent, label="", color=:red, lw=0.5, seriesalpha=0.3)
    end
    Plots.plot!(p_rent, avg_trend_df.Year, avg_trend_df.mean_rent, label="Media 20 Seed", color=:black, lw=3)

    for city_col in names(real_data_df, r"_income")
        Plots.plot!(p_income, anni_reali, real_data_df[!, city_col], label=replace(string(city_col), r"indexed_|_income" => ""), linestyle=:dash, lw=1.5)
    end
    for group in groupby(processed_runs_df, :seed)
        Plots.plot!(p_income, group.Year, group.indexed_income, label="", color=:blue, lw=0.5, seriesalpha=0.3)
    end
    Plots.plot!(p_income, avg_trend_df.Year, avg_trend_df.mean_income, label="Media 20 Seed", color=:black, lw=3)

    final_plot = Plots.plot(p_rent, p_income, layout=(2,1), size=(1200, 1000), legend=:outerright, plot_title="Verifica dei Seed Migliori (Logica Corretta)")
    display(final_plot)
end

# Sostituisci la vecchia funzione con questa nel tuo file di helper o in una cella
function plot_best_seeds_verification_new_style(
    simulation_runs_df::DataFrame, 
    real_data_df::DataFrame,
    scenario_title::String
)
    println("--- Generazione grafico di verifica per lo scenario: $scenario_title ---")

    # --- 1. Processamento Dati (Logica Verificata) ---
    # Inizializza un DataFrame vuoto per raccogliere i dati processati di ogni seed
    processed_runs_df = DataFrame(seed=Int[], Year=Int[], indexed_rent=Float64[], indexed_income=Float64[])
    
    steps_per_year = 8
    n_anni_da_mappare = 10
    window_size_in_steps = 5 * steps_per_year

    # Itera su ogni gruppo di dati corrispondente a un singolo seed
    for group in groupby(simulation_runs_df, :seed)
        df_seed = DataFrame(group)
        
        raw_rent_dyn = df_seed[df_seed.step .> 15, :median_inner_city_rent]
        raw_income_dyn = df_seed[df_seed.step .> 15, :median_income_inner_city]
        
        # Salta il seed se non ha abbastanza dati per la rolling window
        needed_steps = window_size_in_steps + (n_anni_da_mappare - 1) * steps_per_year
        if length(raw_rent_dyn) < needed_steps
            println("ATTENZIONE: Dati insufficienti per il Seed $(first(df_seed.seed)). Saltato.")
            continue
        end

        # Calcola la media mobile e indicizza i dati
        rent_rolling_means = rolling(mean, raw_rent_dyn, window_size_in_steps)
        income_rolling_means = rolling(mean, raw_income_dyn, window_size_in_steps)
        indices_to_sample = [1 + (i - 1) * steps_per_year for i in 1:n_anni_da_mappare]
        sampled_rent_ma = rent_rolling_means[indices_to_sample]
        sampled_income_ma = income_rolling_means[indices_to_sample]
        indexed_sim_rent = (sampled_rent_ma ./ sampled_rent_ma[1]) .* 100 .-100
        indexed_sim_income = (sampled_income_ma ./ sampled_income_ma[1]) .* 100 .-100
        
        anni_da_plottare = minimum(real_data_df.Year):(minimum(real_data_df.Year) + n_anni_da_mappare - 1)
        
        temp_df = DataFrame(seed=first(df_seed.seed), Year=anni_da_plottare, indexed_rent=indexed_sim_rent, indexed_income=indexed_sim_income)
        append!(processed_runs_df, temp_df)
    end

    if isempty(processed_runs_df)
        println("ATTENZIONE: Nessun dato valido da plottare dopo il processamento.")
        return
    end

    # --- 2. Creazione Grafici con il Nuovo Stile ---
    p_rent = Plots.plot(#=title="Affitti: $scenario_title",=# xlabel="Year", ylabel="Rent Percentage Growth since 2011", legend=:outertopright)
    p_income = Plots.plot(#=title="Redditi: $scenario_title",=# xlabel="Year", ylabel="Income Percentage Growth since 2011", legend=:outertopright)

    # Dati reali in secondo piano
    for city_col in names(real_data_df, r"_rent"); Plots.plot!(p_rent, real_data_df.Year, real_data_df[!, city_col], label="", color=:gray, linestyle=:dash, alpha=0.7); end
    Plots.plot!(p_rent, [], [], label="Dati Reali", color=:gray, linestyle=:dash)
    for city_col in names(real_data_df, r"_income"); Plots.plot!(p_income, real_data_df.Year, real_data_df[!, city_col], label="", color=:gray, linestyle=:dash, alpha=0.7); end
    Plots.plot!(p_income, [], [], label="Dati Reali", color=:gray, linestyle=:dash)
    
    # Simulazioni in primo piano, una per colore
    sim_colors = palette(:default)
    for (i, group) in enumerate(groupby(processed_runs_df, :seed))
        seed_id = first(group).seed
        color_idx = mod1(i, length(sim_colors))
        
        Plots.plot!(p_rent, group.Year, group.indexed_rent, label="Seed $seed_id", color=sim_colors[color_idx], lw=2)
        Plots.plot!(p_income, group.Year, group.indexed_income, label="", color=sim_colors[color_idx], lw=2)
    end
    
    # 3. Mostra il risultato finale
    final_plot = Plots.plot(p_rent, p_income, layout=(2,1), size=(1000, 1000))
    display(final_plot)
end

function aggregate_run_data(df::DataFrame)
    mean_nonan(x) = isempty(filter(!isnan, x)) ? NaN : mean(filter(!isnan, x))
    sem_nonan(x) = begin; filtered_x = filter(!isnan, x); n = length(filtered_x); n > 1 ? std(filtered_x) / sqrt(n) : 0.0; end

    summary_df = combine(groupby(df, :step),
        # Aggiungi qui tutte le colonne che vuoi analizzare
        :median_inner_city_rent => mean_nonan => :mean_median_inner_city_rent,
        :median_inner_city_rent => sem_nonan  => :sem_median_inner_city_rent,
        :median_income_inner_city => mean_nonan => :mean_median_income_inner_city,
        :median_income_inner_city => sem_nonan  => :sem_median_income_inner_city,
        :neighbour_segregation_index => mean_nonan => :mean_neighbour_segregation_index,
        :neighbour_segregation_index => sem_nonan  => :sem_neighbour_segregation_index,
        :amenity_segregation_index_low => mean_nonan => :mean_amenity_segregation_index_low,
        :amenity_segregation_index_low => sem_nonan  => :sem_amenity_segregation_index_low,
        :amenity_segregation_index_high => mean_nonan => :mean_amenity_segregation_index_high,
        :amenity_segregation_index_high => sem_nonan  => :sem_amenity_segregation_index_high,
        :n_low_inner => mean => :mean_n_low_inner,
        :n_low_inner => (x -> std(x) / sqrt(length(x)))  => :sem_n_low_inner,
        :n_high_inner => mean => :mean_n_high_inner,
        :n_high_inner => (x -> std(x) / sqrt(length(x)))  => :sem_n_high_inner
    )
    return summary_df
end

function calculate_r_squared(fit_result, y_data)
    # Somma dei quadrati dei residui (distanza tra punti e retta)
    ss_res = sum(fit_result.resid .^ 2)
    # Somma totale dei quadrati (distanza tra punti e la loro media)
    ss_tot = sum((y_data .- mean(y_data)).^2)
    
    # Calcolo di R²
    return 1 - (ss_res / ss_tot)
end


function calculate_regression_slope(df_group::AbstractDataFrame)
    linear_model(x, p) = p[1] .+ p[2] .* x
    
    # Dati per HighIncome
    high_x = skipnan(df_group.mean_quadrant_amenity)
    high_y = skipnan(df_group.prop_high_in_quadrant)
    slope_high = if length(high_x) > 1 && length(high_x) == length(high_y)
        LsqFit.curve_fit(linear_model, high_x, high_y, [0.5, 0.0]).param[2]
    else
        NaN
    end

    # Dati per LowIncome
    low_x = skipnan(df_group.mean_quadrant_amenity)
    low_y = skipnan(df_group.prop_low_in_quadrant)
    slope_low = if length(low_x) > 1 && length(low_x) == length(low_y)
        LsqFit.curve_fit(linear_model, low_x, low_y, [0.5, 0.0]).param[2]
    else
        NaN
    end
    
    return (slope_high=slope_high, slope_low=slope_low)
end


function add_quadrant_proportions!(df::DataFrame)
    # Calcola il totale e le proporzioni solo se non esistono già
    if !("prop_high_in_quadrant" in names(df))
        df.total_agents_in_quadrant = df.n_low_quadrant .+ df.n_high_quadrant
        df.prop_low_in_quadrant = [t > 0 ? n / t : 0.0 for (n, t) in zip(df.n_low_quadrant, df.total_agents_in_quadrant)]
        df.prop_high_in_quadrant = [t > 0 ? n / t : 0.0 for (n, t) in zip(df.n_high_quadrant, df.total_agents_in_quadrant)]
    end
    return df
end

function calculate_regression_metrics(df_group::AbstractDataFrame)
    linear_model(x, p) = p[1] .+ p[2] .* x
    
    # --- High Income ---
    high_x = skipnan(df_group.mean_quadrant_amenity)
    high_y = skipnan(df_group.prop_high_in_quadrant)
    slope_high, r2_high = if length(high_x) > 1 && length(high_x) == length(high_y)
        fit = LsqFit.curve_fit(linear_model, high_x, high_y, [0.5, 0.0])
        pendenza = fit.param[2]
        r_quadro = calculate_r_squared(fit, high_y)
        (pendenza, r_quadro) # Restituisce una tupla
    else
        (NaN, NaN)
    end

    # --- Low Income ---
    low_x = skipnan(df_group.mean_quadrant_amenity)
    low_y = skipnan(df_group.prop_low_in_quadrant)
    slope_low, r2_low = if length(low_x) > 1 && length(low_x) == length(low_y)
        fit = LsqFit.curve_fit(linear_model, low_x, low_y, [0.5, 0.0])
        pendenza = fit.param[2]
        r_quadro = calculate_r_squared(fit, low_y)
        (pendenza, r_quadro) # Restituisce una tupla
    else
        (NaN, NaN)
    end
    
    # Restituisce un NamedTuple con 4 valori
    return (slope_high=slope_high, r2_high=r2_high, slope_low=slope_low, r2_low=r2_low)
end

function calculate_dissimilarity_index(quadrant_df_step, model_df_step)
    total_low = first(model_df_step.n_low_inner) + first(model_df_step.n_low_suburbs)
    total_high = first(model_df_step.n_high_inner) + first(model_df_step.n_high_suburbs)
    if total_low == 0 || total_high == 0; return 0.0; end
    sum_of_differences = 0.0
    for row in eachrow(quadrant_df_step)
        sum_of_differences += abs((row.n_low_quadrant / total_low) - (row.n_high_quadrant / total_high))
    end
    return 0.5 * sum_of_differences
end

function calculate_dissimilarity_index_pairwise(quadrant_df_step, model_df_step;
                                                group1_col::Symbol,
                                                group2_col::Symbol,
                                                total_group1::Union{Nothing, Int}=nothing,
                                                total_group2::Union{Nothing, Int}=nothing)
    total_g1 = isnothing(total_group1) ? sum(quadrant_df_step[!, group1_col]) : total_group1
    total_g2 = isnothing(total_group2) ? sum(quadrant_df_step[!, group2_col]) : total_group2

    if total_g1 == 0 || total_g2 == 0
        return 0.0
    end

    sum_of_differences = sum(abs((row[group1_col]/total_g1) - (row[group2_col]/total_g2))
                             for row in eachrow(quadrant_df_step))

    return 0.5 * sum_of_differences
end

function calculate_quadrant_data_per_step(model_to_sample, current_step_number)

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
            current_step_quadrant_counts[quadrant_coords] = Dict(:n_low => 0, :n_medium => 0, :n_high => 0)
        end

        if agent_obj isa LowIncome
            current_step_quadrant_counts[quadrant_coords][:n_low] += 1
        elseif agent_obj isa HighIncome
            if agent_obj.income_value <= model_to_sample.medium_high_income_cutoff
                current_step_quadrant_counts[quadrant_coords][:n_medium] += 1
            else
                current_step_quadrant_counts[quadrant_coords][:n_high] += 1
            end
        end
    end

    all_quadrant_keys = sort(collect(keys(current_step_quadrant_counts)))
    
    for q_coords in all_quadrant_keys
        n_low = get(current_step_quadrant_counts, q_coords, Dict(:n_low => 0))[:n_low]
        n_medium = get(current_step_quadrant_counts, q_coords, Dict(:n_medium => 0))[:n_medium]
        n_high = get(current_step_quadrant_counts, q_coords, Dict(:n_high => 0))[:n_high]
        

        push!(history_quadrant_step, current_step_number)
        push!(history_quadrant_coords_x, q_coords[1])
        push!(history_quadrant_coords_y, q_coords[2])
        push!(history_quadrant_n_low, n_low)
        push!(history_quadrant_n_medium, n_medium)
        push!(history_quadrant_n_high, n_high)
    end

end
using DataFrames
using Statistics
using Printf

function calculate_entropy(x, y, z)
    tot = x + y + z
    if tot == 0
        return 0.0
    end

    probs = [x, y, z] ./ tot
    entropy = 0.0
    for p in probs
        if p > 0  # evita ln(0)
            entropy += p * log(1/p)
        end
    end
    return entropy
end

function calculate_theil_step(df_step)
    # Totali globali
    total_low = sum(df_step[:, :n_low_quadrant])
    total_medium = sum(df_step[:, :n_medium_quadrant])
    total_high = sum(df_step[:, :n_high_quadrant])

    T = total_low + total_medium + total_high
    if T == 0
        return 0.0
    end

    # Entropia totale
    E = calculate_entropy(total_low, total_medium, total_high)

    # Calcolo indice H
    H = 0.0
    for row in eachrow(df_step)
        t = row.n_low_quadrant + row.n_medium_quadrant + row.n_high_quadrant
        if t > 0
            e = calculate_entropy(row.n_low_quadrant, row.n_medium_quadrant, row.n_high_quadrant)
            H += t * (E - e)
        end
    end

    H /= (E * T)
    return H
end

function calculate_theil_decomposed(gdf)
    # --- 1. Identifica le zone ---
    # Definiamo i quadranti dell'inner city (assumendo griglia 4x4)
    inner_city_coords = [(2,2), (2,3), (3,2), (3,3)]
    
    # Aggiungiamo una colonna per identificare la zona di ogni quadrante
    gdf.zone = map(eachrow(gdf)) do row
        (row.quadrant_coords_x, row.quadrant_coords_y) in inner_city_coords ? :inner : :suburb
    end

    df_inner = filter(row -> row.zone == :inner, gdf)
    df_suburb = filter(row -> row.zone == :suburb, gdf)

    # --- 2. Calcola H Totale (riutilizzando la tua funzione) ---
    H_total = calculate_theil_step(gdf)

    # --- 3. Calcola H Intra-zona (Within) per ciascuna zona ---
    H_within_inner = calculate_theil_step(df_inner)
    H_within_suburb = calculate_theil_step(df_suburb)

    # --- 4. Calcola H Inter-zona (Between) ---
    # Totali globali
    T_total_low = sum(gdf.n_low_quadrant)
    T_total_medium = sum(gdf.n_medium_quadrant)
    T_total_high = sum(gdf.n_high_quadrant)
    T_total = T_total_low + T_total_medium + T_total_high

    if T_total == 0; return (H_total=0.0, H_between=0.0, H_within_inner=0.0, H_within_suburb=0.0); end
    
    E_total = calculate_entropy(T_total_low, T_total_medium, T_total_high)
    if E_total == 0; return (H_total=0.0, H_between=0.0, H_within_inner=0.0, H_within_suburb=0.0); end

    # Totali per zona
    t_inner = sum(df_inner.n_low_quadrant) + sum(df_inner.n_medium_quadrant) + sum(df_inner.n_high_quadrant)
    t_suburb = sum(df_suburb.n_low_quadrant) + sum(df_suburb.n_medium_quadrant) + sum(df_suburb.n_high_quadrant)

    # Entropia di ciascuna zona
    E_inner = calculate_entropy(sum(df_inner.n_low_quadrant), sum(df_inner.n_medium_quadrant), sum(df_inner.n_high_quadrant))
    E_suburb = calculate_entropy(sum(df_suburb.n_low_quadrant), sum(df_suburb.n_medium_quadrant), sum(df_suburb.n_high_quadrant))

    # Entropia media ponderata "within-zone"
    E_avg_within_zone = 0.0
    if T_total > 0
        E_avg_within_zone = (t_inner / T_total) * E_inner + (t_suburb / T_total) * E_suburb
    end
    
    H_between = (E_total - E_avg_within_zone) / E_total
    
    return (
        H_total = H_total,
        H_between = H_between,
        H_within_inner = H_within_inner,
        H_within_suburb = H_within_suburb
    )
end

function calculate_exposure_indices(quadrant_df_step, model_df_step)
    total_low = first(model_df_step.n_low_inner) + first(model_df_step.n_low_suburbs)
    total_high = first(model_df_step.n_high_inner) + first(model_df_step.n_high_suburbs)
    total_population = total_low + total_high
    if total_low == 0 || total_high == 0 || total_population == 0; return (exposure_L_H=NaN, isolation_L_L=NaN, isolation_H_H=NaN); end

    exposure_low_to_high, isolation_low, isolation_high = 0.0, 0.0, 0.0
    for row in eachrow(quadrant_df_step)
        n_low_q, n_high_q = row.n_low_quadrant, row.n_high_quadrant
        total_q = n_low_q + n_high_q
        if total_q > 0
            exposure_low_to_high += (n_low_q / total_low) * (n_high_q / total_q)
            isolation_low += (n_low_q / total_low) * (n_low_q / total_q)
            isolation_high += (n_high_q / total_high) * (n_high_q / total_q)
        end
    end

    r_high = total_high / total_population
    r_low = total_low / total_population
    
    # Restituisce un NamedTuple, così possiamo accedere ai campi con .nome
    return (
        exposure_L_H = r_high > 0 ? exposure_low_to_high / r_high : 0.0, 
        isolation_L_L = r_low > 0 ? isolation_low / r_low : 0.0,
        isolation_H_H = r_high > 0 ? isolation_high / r_high : 0.0
    )
end




function plot_percolation_map(model; plot_title_prefix="Mappa di Percolazione")
    
    # --- MODIFICA: Chiama le funzioni helper invece di ricalcolare ---
    
    # 1. Chiama la funzione per calcolare la dimensione del cluster
    largest_cluster_size = calculate_largest_blocking_cluster(model)
    
    # 2. Chiama la funzione per ottenere le case con spillover per la colorazione
    spillover_houses = get_spillover_houses(model)

    # --- Creazione del grafico ---
    plot_title = "$(plot_title_prefix)\nCluster Max: $(largest_cluster_size) patch"
    dims = size(model.space)
    
    p = Plots.plot(title=plot_title, xlims=(0, dims[2]+1), ylims=(0, dims[1]+1), 
             aspect_ratio=:equal, legend=:outertopright, axis=false, grid=false, background_color=:gainsboro)

    for r in 1:model.grid_dims[1], c in 1:model.grid_dims[2]
        pos = (r, c)
        type = model.patch_type[pos...]
        y_pos = model.grid_dims[1] - r + 1 
        
        local color_to_use
        if type == :cbd
            color_to_use = :black
        elseif type == :public
            color_to_use = :green
        elseif type == :house
            mgmt = model.management_type[pos...]
            if mgmt == :public
                color_to_use = :red
            elseif pos in spillover_houses
                color_to_use = :pink
            else
                continue 
            end
        else
            continue
        end
        
        Plots.scatter!(p, [c], [y_pos], marker=:square, color=color_to_use, label="", markersize=4.5)
    end
    
    # Legenda "dummy"
    Plots.scatter!(p, [], [], marker=:square, color=:green, label="Amenity")
    Plots.scatter!(p, [], [], marker=:square, color=:black, label="CBD")
    Plots.scatter!(p, [], [], marker=:square, color=:red, label="Public Housing")
    Plots.scatter!(p, [], [], marker=:square, color=:pink, label="Spillover Zone")

    display(p)
end

function plot_percolation_map_von_neumann(model; plot_title_prefix="Mappa di Percolazione")
    
    # --- MODIFICA: Chiama le funzioni helper invece di ricalcolare ---
    
    # 1. Chiama la funzione per calcolare la dimensione del cluster
    largest_cluster_size = calculate_largest_cluster_von_neumann(model)
    
    # 2. Chiama la funzione per ottenere le case con spillover per la colorazione
    spillover_houses = get_spillover_houses(model)

    # --- Creazione del grafico ---
    plot_title = "$(plot_title_prefix)\nCluster Max: $(largest_cluster_size) patch"
    dims = size(model.space)
    
    p = Plots.plot(title=plot_title, xlims=(0, dims[2]+1), ylims=(0, dims[1]+1), 
             aspect_ratio=:equal, legend=:outertopright, axis=false, grid=false, background_color=:gainsboro)

    for r in 1:model.grid_dims[1], c in 1:model.grid_dims[2]
        pos = (r, c)
        type = model.patch_type[pos...]
        y_pos = model.grid_dims[1] - r + 1 
        
        local color_to_use
        if type == :cbd
            color_to_use = :black
        elseif type == :public
            color_to_use = :green
        elseif type == :house
            mgmt = model.management_type[pos...]
            if mgmt == :public
                color_to_use = :red
            elseif pos in spillover_houses
                color_to_use = :pink
            else
                continue 
            end
        else
            continue
        end
        
        Plots.scatter!(p, [c], [y_pos], marker=:square, color=color_to_use, label="", markersize=4.5)
    end
    
    # Legenda "dummy"
    Plots.scatter!(p, [], [], marker=:square, color=:green, label="Amenity")
    Plots.scatter!(p, [], [], marker=:square, color=:black, label="CBD")
    Plots.scatter!(p, [], [], marker=:square, color=:red, label="Public Housing")
    Plots.scatter!(p, [], [], marker=:square, color=:pink, label="Spillover Zone")

    display(p)
end


function calculate_largest_blocking_cluster(model)
    
    # --- 1. Identifica tutte le patch che fanno parte dei cluster ---
    
    # a. Trova le case pubbliche
    public_housing_patches = findall(m_type -> m_type == :public, model.management_type)
    
    # b. Calcola le case private nella zona di spillover
    spillover_houses = Set{NTuple{2, Int}}()
    if model.public_housing_active
        for pos_ph_cartesian in public_housing_patches
            pos_ph = Tuple(pos_ph_cartesian)
            # Usiamo il raggio di spillover definito nel modello
            neighbor_positions = nearby_positions(pos_ph, model, model.public_housing_conversion_radius)
            for neighbor_pos in neighbor_positions
                if model.patch_type[neighbor_pos...] == :house && 
                   model.management_type[neighbor_pos...] == :private &&
                   !(neighbor_pos in model.renovated_houses_history)
                    push!(spillover_houses, neighbor_pos)
                end
            end
        end
    end

    # c. Unisci tutte le patch di "blocco"
    blocking_patches = Set{NTuple{2, Int}}()
    # Aggiungi le case pubbliche (convertendo da CartesianIndex a NTuple)
    for pos_cartesian in public_housing_patches
        push!(blocking_patches, Tuple(pos_cartesian))
    end
    # Aggiungi le case con spillover
    union!(blocking_patches, spillover_houses)
    # Aggiungi le aree verdi
    for pos_cartesian in findall(p_type -> p_type == :public, model.patch_type)
        push!(blocking_patches, Tuple(pos_cartesian))
    end

    if isempty(blocking_patches)
        return 0
    end
    
    # --- 2. Costruisci il grafo e trova i cluster ---
    
    # Mappa le coordinate delle patch a un indice intero per il grafo
    patch_to_vertex = Dict(pos => i for (i, pos) in enumerate(blocking_patches))
    g = SimpleGraph(length(blocking_patches))

    # Aggiungi gli archi tra patch vicine
    for (pos, v_idx) in patch_to_vertex
        # Quartiere di Moore (8 direzioni)
        for dx in -1:1, dy in -1:1
            if dx == 0 && dy == 0; continue; end
            neighbor_pos = (pos[1] + dx, pos[2] + dy)
            
            # Se anche il vicino è una patch di blocco, aggiungi un arco
            if haskey(patch_to_vertex, neighbor_pos)
                neighbor_v_idx = patch_to_vertex[neighbor_pos]
                add_edge!(g, v_idx, neighbor_v_idx)
            end
        end
    end
    
    # --- 3. Calcola le dimensioni dei cluster e restituisci la più grande ---
    
    # Trova tutti i gruppi di vertici connessi (i nostri cluster)
    components = connected_components(g)
    
    if isempty(components)
        return 0
    end
    
    # Restituisci la dimensione (numero di patch) del cluster più grande
    return maximum(length.(components))
end

function calculate_largest_cluster_von_neumann(model)
    
    # --- 1. Identifica tutte le patch che fanno parte dei cluster ---
    
    # a. Trova le case pubbliche
    public_housing_patches = findall(m_type -> m_type == :public, model.management_type)
    
    # b. Calcola le case private nella zona di spillover
    spillover_houses = Set{NTuple{2, Int}}()
    if model.public_housing_active
        for pos_ph_cartesian in public_housing_patches
            pos_ph = Tuple(pos_ph_cartesian)
            # Usiamo il raggio di spillover definito nel modello
            neighbor_positions = nearby_positions(pos_ph, model, model.public_housing_conversion_radius)
            for neighbor_pos in neighbor_positions
                if model.patch_type[neighbor_pos...] == :house && 
                   model.management_type[neighbor_pos...] == :private &&
                   !(neighbor_pos in model.renovated_houses_history)
                    push!(spillover_houses, neighbor_pos)
                end
            end
        end
    end

    # c. Unisci tutte le patch di "blocco"
    blocking_patches = Set{NTuple{2, Int}}()
    # Aggiungi le case pubbliche (convertendo da CartesianIndex a NTuple)
    for pos_cartesian in public_housing_patches
        push!(blocking_patches, Tuple(pos_cartesian))
    end
    # Aggiungi le case con spillover
    union!(blocking_patches, spillover_houses)
    # Aggiungi le aree verdi
    for pos_cartesian in findall(p_type -> p_type == :public, model.patch_type)
        push!(blocking_patches, Tuple(pos_cartesian))
    end

    if isempty(blocking_patches)
        return 0
    end
    
    # --- 2. Costruisci il grafo e trova i cluster ---
    patch_to_vertex = Dict(pos => i for (i, pos) in enumerate(blocking_patches))
    g = SimpleGraph(length(blocking_patches))

    # --- MODIFICA CHIAVE: Ciclo sui vicini ---
    # Definiamo esplicitamente le 4 direzioni (su, giù, destra, sinistra)
    neighbor_offsets = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    for (pos, v_idx) in patch_to_vertex
        for (dx, dy) in neighbor_offsets
            neighbor_pos = (pos[1] + dx, pos[2] + dy)
            
            # Se anche il vicino è una patch di blocco, aggiungi un arco
            if haskey(patch_to_vertex, neighbor_pos)
                neighbor_v_idx = patch_to_vertex[neighbor_pos]
                add_edge!(g, v_idx, neighbor_v_idx)
            end
        end
    end
    
    # --- 3. Calcola le dimensioni dei cluster (invariato) ---
    components = connected_components(g)
    if isempty(components); return 0; end
    
    return maximum(length.(components))
end


function calculate_average_cluster_size_von_neumann(model)
    
    # --- 1. Identifica tutte le patch che fanno parte dei cluster ---
    
    # a. Trova le case pubbliche
    public_housing_patches = findall(m_type -> m_type == :public, model.management_type)
    
    # b. Calcola le case private nella zona di spillover
    spillover_houses = Set{NTuple{2, Int}}()
    if model.public_housing_active
        for pos_ph_cartesian in public_housing_patches
            pos_ph = Tuple(pos_ph_cartesian)
            # Usiamo il raggio di spillover definito nel modello
            neighbor_positions = nearby_positions(pos_ph, model, model.public_housing_conversion_radius)
            for neighbor_pos in neighbor_positions
                if model.patch_type[neighbor_pos...] == :house && 
                   model.management_type[neighbor_pos...] == :private &&
                   !(neighbor_pos in model.renovated_houses_history)
                    push!(spillover_houses, neighbor_pos)
                end
            end
        end
    end

    # c. Unisci tutte le patch di "blocco"
    blocking_patches = Set{NTuple{2, Int}}()
    # Aggiungi le case pubbliche (convertendo da CartesianIndex a NTuple)
    for pos_cartesian in public_housing_patches
        push!(blocking_patches, Tuple(pos_cartesian))
    end
    # Aggiungi le case con spillover
    union!(blocking_patches, spillover_houses)
    # Aggiungi le aree verdi
    for pos_cartesian in findall(p_type -> p_type == :public, model.patch_type)
        push!(blocking_patches, Tuple(pos_cartesian))
    end

    if isempty(blocking_patches)
        return 0
    end
    
    # --- 2. Costruisci il grafo e trova i cluster ---
    patch_to_vertex = Dict(pos => i for (i, pos) in enumerate(blocking_patches))
    g = SimpleGraph(length(blocking_patches))

    # --- MODIFICA CHIAVE: Ciclo sui vicini ---
    # Definiamo esplicitamente le 4 direzioni (su, giù, destra, sinistra)
    neighbor_offsets = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    for (pos, v_idx) in patch_to_vertex
        for (dx, dy) in neighbor_offsets
            neighbor_pos = (pos[1] + dx, pos[2] + dy)
            
            # Se anche il vicino è una patch di blocco, aggiungi un arco
            if haskey(patch_to_vertex, neighbor_pos)
                neighbor_v_idx = patch_to_vertex[neighbor_pos]
                add_edge!(g, v_idx, neighbor_v_idx)
            end
        end
    end
    
    # --- 3. Calcola le dimensioni dei cluster (invariato) ---
    components = connected_components(g)
    if isempty(components); return 0; end
    
    # --- MODIFICA CHIAVE ---
    # Calcola la dimensione di ogni singolo cluster
    cluster_sizes = length.(components)
    
    # Restituisci la MEDIA delle dimensioni, non il massimo
    return mean(cluster_sizes)
end

function calculate_largest_inner_city_cluster_von_neumann(model)
    
    # --- 1. Identifica tutte le patch di blocco sull'intera mappa (come prima) ---
    public_housing_patches = findall(m_type -> m_type == :public, model.management_type)
    spillover_houses = get_spillover_houses(model) # Usiamo la funzione helper che abbiamo definito
    
    all_blocking_patches = Set{NTuple{2, Int}}()
    for pos_cartesian in public_housing_patches; push!(all_blocking_patches, Tuple(pos_cartesian)); end
    union!(all_blocking_patches, spillover_houses)
    for pos_cartesian in findall(p_type -> p_type == :public, model.patch_type); push!(all_blocking_patches, Tuple(pos_cartesian)); end

    # --- MODIFICA CHIAVE: Filtra le patch per tenere solo quelle nell'inner city ---
    inner_city_blocking_patches = Set{NTuple{2, Int}}()
    for pos in all_blocking_patches
        if is_in_inner_city(pos, model)
            push!(inner_city_blocking_patches, pos)
        end
    end

    if isempty(inner_city_blocking_patches)
        return 0
    end
    
    # --- 2. Costruisci il grafo e trova i cluster usando solo le patch dell'inner city ---
    patch_to_vertex = Dict(pos => i for (i, pos) in enumerate(inner_city_blocking_patches))
    g = SimpleGraph(length(inner_city_blocking_patches))
    
    neighbor_offsets = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    for (pos, v_idx) in patch_to_vertex
        for (dx, dy) in neighbor_offsets
            neighbor_pos = (pos[1] + dx, pos[2] + dy)
            if haskey(patch_to_vertex, neighbor_pos)
                add_edge!(g, v_idx, patch_to_vertex[neighbor_pos])
            end
        end
    end
    
    # --- 3. Calcola il risultato (invariato) ---
    components = connected_components(g)
    if isempty(components); return 0; end
    
    return maximum(length.(components))
end

function calculate_largest_inner_city_cluster_moore(model)
    
    # --- 1. Identifica tutte le patch di blocco sull'intera mappa (come prima) ---
    public_housing_patches = findall(m_type -> m_type == :public, model.management_type)
    spillover_houses = get_spillover_houses(model) # Usiamo la funzione helper che abbiamo definito
    
    all_blocking_patches = Set{NTuple{2, Int}}()
    for pos_cartesian in public_housing_patches; push!(all_blocking_patches, Tuple(pos_cartesian)); end
    union!(all_blocking_patches, spillover_houses)
    for pos_cartesian in findall(p_type -> p_type == :public, model.patch_type); push!(all_blocking_patches, Tuple(pos_cartesian)); end

    # --- MODIFICA CHIAVE: Filtra le patch per tenere solo quelle nell'inner city ---
    inner_city_blocking_patches = Set{NTuple{2, Int}}()
    for pos in all_blocking_patches
        if is_in_inner_city(pos, model)
            push!(inner_city_blocking_patches, pos)
        end
    end

    if isempty(inner_city_blocking_patches)
        return 0
    end
    
    # --- 2. Costruisci il grafo e trova i cluster usando solo le patch dell'inner city ---
    patch_to_vertex = Dict(pos => i for (i, pos) in enumerate(inner_city_blocking_patches))
    g = SimpleGraph(length(inner_city_blocking_patches))
    
    for (pos, v_idx) in patch_to_vertex
        # Itera su tutte le 8 direzioni circostanti (inclusi gli angoli)
        for dx in -1:1, dy in -1:1
            # Salta la patch centrale (se stessa)
            if dx == 0 && dy == 0; continue; end
            
            neighbor_pos = (pos[1] + dx, pos[2] + dy)
            
            # Se anche il vicino è una patch di blocco, aggiungi un arco
            if haskey(patch_to_vertex, neighbor_pos)
                neighbor_v_idx = patch_to_vertex[neighbor_pos]
                add_edge!(g, v_idx, neighbor_v_idx)
            end
        end
    end
    
    # --- 3. Calcola il risultato (invariato) ---
    components = connected_components(g)
    if isempty(components); return 0; end
    
    return maximum(length.(components))
end


function calculate_number_of_clusters_von_neumann(model)
    
    # --- 1. Identifica tutte le patch che fanno parte dei cluster ---
    
    # a. Trova le case pubbliche
    public_housing_patches = findall(m_type -> m_type == :public, model.management_type)
    
    # b. Calcola le case private nella zona di spillover
    spillover_houses = Set{NTuple{2, Int}}()
    if model.public_housing_active
        for pos_ph_cartesian in public_housing_patches
            pos_ph = Tuple(pos_ph_cartesian)
            # Usiamo il raggio di spillover definito nel modello
            neighbor_positions = nearby_positions(pos_ph, model, model.public_housing_conversion_radius)
            for neighbor_pos in neighbor_positions
                if model.patch_type[neighbor_pos...] == :house && 
                   model.management_type[neighbor_pos...] == :private &&
                   !(neighbor_pos in model.renovated_houses_history)
                    push!(spillover_houses, neighbor_pos)
                end
            end
        end
    end

    # c. Unisci tutte le patch di "blocco"
    blocking_patches = Set{NTuple{2, Int}}()
    # Aggiungi le case pubbliche (convertendo da CartesianIndex a NTuple)
    for pos_cartesian in public_housing_patches
        push!(blocking_patches, Tuple(pos_cartesian))
    end
    # Aggiungi le case con spillover
    #union!(blocking_patches, spillover_houses)
    # Aggiungi le aree verdi
    for pos_cartesian in findall(p_type -> p_type == :public, model.patch_type)
        push!(blocking_patches, Tuple(pos_cartesian))
    end

    if isempty(blocking_patches)
        return 0
    end
    
    # --- 2. Costruisci il grafo e trova i cluster ---
    patch_to_vertex = Dict(pos => i for (i, pos) in enumerate(blocking_patches))
    g = SimpleGraph(length(blocking_patches))

    # --- MODIFICA CHIAVE: Ciclo sui vicini ---
    # Definiamo esplicitamente le 4 direzioni (su, giù, destra, sinistra)
    neighbor_offsets = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    for (pos, v_idx) in patch_to_vertex
        for (dx, dy) in neighbor_offsets
            neighbor_pos = (pos[1] + dx, pos[2] + dy)
            
            # Se anche il vicino è una patch di blocco, aggiungi un arco
            if haskey(patch_to_vertex, neighbor_pos)
                neighbor_v_idx = patch_to_vertex[neighbor_pos]
                add_edge!(g, v_idx, neighbor_v_idx)
            end
        end
    end
    
    # --- 3. Calcola le dimensioni dei cluster (invariato) ---
    components = connected_components(g)
    return length(components)
end



function calculate_interface_length(model)
    
    # --- 1. Identifica tutte le patch che fanno parte dei cluster ---
    
    # a. Trova le case pubbliche
    public_housing_patches = findall(m_type -> m_type == :public, model.management_type)
    
    # b. Calcola le case private nella zona di spillover
    spillover_houses = Set{NTuple{2, Int}}()
    if model.public_housing_active
        for pos_ph_cartesian in public_housing_patches
            pos_ph = Tuple(pos_ph_cartesian)
            # Usiamo il raggio di spillover definito nel modello
            neighbor_positions = nearby_positions(pos_ph, model, model.public_housing_conversion_radius)
            for neighbor_pos in neighbor_positions
                if model.patch_type[neighbor_pos...] == :house && 
                   model.management_type[neighbor_pos...] == :private &&
                   !(neighbor_pos in model.renovated_houses_history)
                    push!(spillover_houses, neighbor_pos)
                end
            end
        end
    end

    # c. Unisci tutte le patch di "blocco"
    blocking_patches = Set{NTuple{2, Int}}()
    # Aggiungi le case pubbliche (convertendo da CartesianIndex a NTuple)
    for pos_cartesian in public_housing_patches
        push!(blocking_patches, Tuple(pos_cartesian))
    end
    # Aggiungi le case con spillover
    #union!(blocking_patches, spillover_houses)
    # Aggiungi le aree verdi
    for pos_cartesian in findall(p_type -> p_type == :public, model.patch_type)
        push!(blocking_patches, Tuple(pos_cartesian))
    end

    if isempty(blocking_patches); return 0; end

    interface_length = 0
    neighbor_offsets = [(0, 1), (0, -1), (1, 0), (-1, 0)] # Vicinato a 4

    # 2. Per ogni patch di blocco, controlla i suoi vicini
    for pos in blocking_patches
        for (dx, dy) in neighbor_offsets
            neighbor_pos = (pos[1] + dx, pos[2] + dy)

            # Controlla se il vicino è DENTRO la griglia
            if 1 <= neighbor_pos[1] <= model.grid_dims[1] && 1 <= neighbor_pos[2] <= model.grid_dims[2]
                # Se il vicino NON è una patch di blocco, allora è un confine!
                if !(neighbor_pos in blocking_patches)
                    interface_length += 1
                end
            end
        end
    end

    return interface_length
end


function get_spillover_houses(model)
    spillover_houses = Set{NTuple{2, Int}}()
    if model.public_housing_active
        public_housing_patches = findall(m_type -> m_type == :public, model.management_type)
        for pos_ph_cartesian in public_housing_patches
            pos_ph = Tuple(pos_ph_cartesian)
            neighbor_positions = nearby_positions(pos_ph, model, model.public_housing_conversion_radius)
            for neighbor_pos in neighbor_positions
                if model.patch_type[neighbor_pos...] == :house && 
                   model.management_type[neighbor_pos...] == :private && 
                   !(neighbor_pos in model.renovated_houses_history)
                    push!(spillover_houses, neighbor_pos)
                end
            end
        end
    end
    return spillover_houses
end


function calculate_spillover_overlap(model)
    
    # 1. Trova le case attualmente soggette a spillover negativo da PH
    # (usiamo la funzione helper che già abbiamo)
    negative_spillover_set = get_spillover_houses(model)
    if isempty(negative_spillover_set)
        return 0
    end

    # 2. Trova tutte le case che hanno MAI subito uno spillover positivo
    #    verificando i vicini di ogni casa mai ristrutturata.
    positive_spillover_zone = Set{NTuple{2, Int}}()
    if hasproperty(model, :renovated_houses_history) && !isempty(model.renovated_houses_history)
        for pos_renovated in model.renovated_houses_history
            # Trova i vicini di ogni casa mai ristrutturata
            neighbor_positions = nearby_positions(pos_renovated, model, model.gentrification_spillover_radius)
            for neighbor_pos in neighbor_positions
                # Aggiungi solo le case, escludendo la casa ristrutturata stessa
                if model.patch_type[neighbor_pos...] == :house && neighbor_pos != pos_renovated
                    push!(positive_spillover_zone, neighbor_pos)
                end
            end
        end
    end
    
    # 3. Calcola l'intersezione tra i due insiemi
    #    `intersect` trova gli elementi comuni a entrambi i Set
    overlap_set = intersect(negative_spillover_set, positive_spillover_zone)
    
    # 4. Restituisci il numero di case nell'intersezione
    return length(overlap_set)
end


function count_inner_city_public_housing(model)
    
    public_housing_count = 0
    
    # Itera su tutte le posizioni della griglia
    for r in 1:model.grid_dims[1], c in 1:model.grid_dims[2]
        pos = (r, c)
        
        # Controlla se la patch è nell'inner city
        if is_in_inner_city(pos, model)
            # Se sì, controlla se è una casa pubblica
            if model.management_type[pos...] == :public
                public_housing_count += 1
            end
        end
    end
    
    return public_housing_count
end


#=
function plot_city_layout(model::ABM)
    # Calcola i limiti fissi e la colormap per l'età delle case
    # --- RECUPERA I LIMITI DAI PARAMETRI DEL MODELLO ---
    min_age_plot, max_age_plot = model.plot_age_range
    min_amenity_plot, max_amenity_plot = model.plot_amenity_range


   # --- INIZIO BLOCCO MANCANTE: DEFINIZIONE COLORMAP E VARIABILI ---
    N_house_age_bins = 20
    N_amenity_bins = 20  
    
    final_colormap = RGBA{Float64}[]
    id_empty = 1
    push!(final_colormap, RGBA(1.0, 1.0, 1.0, 1.0)) # Colore per :empty

    # Colori per l'età delle case
    base_r_yellow = 1.0; base_g_yellow = 0.95; base_b_yellow = 0.7; 
    base_r_orange = 0.9; base_g_orange = 0.45; base_b_orange = 0.0; 
    house_age_sub_colormap = RGBA{Float64}[]
    for i_bin in 0:(N_house_age_bins-1)
        norm_val_for_bin = (N_house_age_bins == 1) ? 0.5 : i_bin / (N_house_age_bins - 1)
        r_val = base_r_yellow + norm_val_for_bin * (base_r_orange - base_r_yellow)
        g_val = base_g_yellow + norm_val_for_bin * (base_g_orange - base_g_yellow)
        b_val = base_b_yellow + norm_val_for_bin * (base_b_orange - base_b_yellow)
        push!(final_colormap, RGBA(r_val, g_val, b_val, 1.0))
        push!(house_age_sub_colormap, RGBA(r_val, g_val, b_val, 1.0))
    end
    offset_after_house = id_empty + N_house_age_bins

    # Colori per le amenità
    base_r_dark_green = 0.1; base_g_dark_green = 0.3; base_b_dark_green = 0.1;
    base_r_light_green = 0.6; base_g_light_green = 1.0; base_b_light_green = 0.6;
    amenity_greens_sub_colormap = RGBA{Float64}[] 
    for i_bin in 0:(N_amenity_bins-1)
        norm_val_for_bin = (N_amenity_bins == 1) ? 0.5 : i_bin / (N_amenity_bins - 1)
        r_val = base_r_dark_green + norm_val_for_bin * (base_r_light_green - base_r_dark_green)
        g_val = base_g_dark_green + norm_val_for_bin * (base_g_light_green - base_g_dark_green)
        b_val = base_b_dark_green + norm_val_for_bin * (base_b_light_green - base_b_dark_green)
        push!(final_colormap, RGBA(r_val, g_val, b_val, 1.0))
        push!(amenity_greens_sub_colormap, RGBA(r_val, g_val, b_val, 1.0))
    end
    offset_after_public = offset_after_house + N_amenity_bins

    # Colore per il CBD
    id_cbd = offset_after_public + 1
    push!(final_colormap, RGBA(0.2, 0.2, 0.2, 1.0)) 
    total_colormap_entries = length(final_colormap)
    # --- FINE BLOCCO MANCANTE ---
    # Crea la matrice degli ID dei colori
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

    # --- Creazione Plot (Logica Modificata) ---
    
    # 1. Crea una figura vuota
    fig = Figure(resolution = (1000, 900))

    # 2. Crea gli assi per la mappa nella prima riga
    ax = Axis(fig[1, 1], 
              aspect = DataAspect() # Mantiene le proporzioni corrette
    )
    
    # 3. Disegna la heatmap sugli assi appena creati
    abmplot!(ax, model;
        agent_size = 0,
        heatarray = _ -> patch_ids_for_colormap,
        heatkwargs = ( colormap = final_colormap, colorrange = (1, total_colormap_entries) ),
        add_colorbar = false
    )

    # 4. Aggiungi il bordo dell'inner city
    r_start, r_end, c_start, c_end = get_inner_city_bounds(model)
    x_coords = [c_start - 0.5, c_end + 0.5, c_end + 0.5, c_start - 0.5, c_start - 0.5]
    y_coords = [r_start - 0.5, r_start - 0.5, r_end + 0.5, r_end + 0.5, r_start - 0.5]
    lines!(ax, x_coords, y_coords, color = :black, linewidth = 2, linestyle = :dash)

    # 5. Aggiungi la Colorbar SOTTO, specificando l'altezza e l'orientamento
    Colorbar(fig[2, 1], # Posizione: Seconda riga, prima colonna
        colormap = house_age_sub_colormap, 
        limits = (min_age_plot, max_age_plot), 
        label = "House Age",
        vertical = false,      # Rendila orizzontale
        height = 30,           # Aumenta lo spessore
        labelsize = 22,
        ticklabelsize = 18,
        flipaxis = false       # Assicura che i valori vadano da sx a dx
    )
    
    # 6. Aggiusta il layout per dare spazio alla colorbar
    rowgap!(fig.layout, 1, 10) # Spazio tra il grafico e la colorbar

    display(fig)
end=#



function plot_city_layout(model::ABM)
    # Calcola i limiti fissi e la colormap per l'età delle case
    # --- RECUPERA I LIMITI DAI PARAMETRI DEL MODELLO ---
    min_age_plot, max_age_plot = model.plot_age_range
    min_amenity_plot, max_amenity_plot = model.plot_amenity_range


   # --- INIZIO BLOCCO MANCANTE: DEFINIZIONE COLORMAP E VARIABILI ---
    N_house_age_bins = 20
    N_amenity_bins = 20  
    
    final_colormap = RGBA{Float64}[]
    id_empty = 1
    push!(final_colormap, RGBA(1.0, 1.0, 1.0, 1.0)) # Colore per :empty

    # Colori per l'età delle case
    base_r_yellow = 1.0; base_g_yellow = 0.95; base_b_yellow = 0.7; 
    base_r_orange = 0.9; base_g_orange = 0.45; base_b_orange = 0.0; 
    house_age_sub_colormap = RGBA{Float64}[]
    for i_bin in 0:(N_house_age_bins-1)
        norm_val_for_bin = (N_house_age_bins == 1) ? 0.5 : i_bin / (N_house_age_bins - 1)
        r_val = base_r_yellow + norm_val_for_bin * (base_r_orange - base_r_yellow)
        g_val = base_g_yellow + norm_val_for_bin * (base_g_orange - base_g_yellow)
        b_val = base_b_yellow + norm_val_for_bin * (base_b_orange - base_b_yellow)
        push!(final_colormap, RGBA(r_val, g_val, b_val, 1.0))
        push!(house_age_sub_colormap, RGBA(r_val, g_val, b_val, 1.0))
    end
    offset_after_house = id_empty + N_house_age_bins

    # Colori per le amenità
    base_r_dark_green = 0.1; base_g_dark_green = 0.3; base_b_dark_green = 0.1;
    base_r_light_green = 0.6; base_g_light_green = 1.0; base_b_light_green = 0.6;
    amenity_greens_sub_colormap = RGBA{Float64}[] 
    for i_bin in 0:(N_amenity_bins-1)
        norm_val_for_bin = (N_amenity_bins == 1) ? 0.5 : i_bin / (N_amenity_bins - 1)
        r_val = base_r_dark_green + norm_val_for_bin * (base_r_light_green - base_r_dark_green)
        g_val = base_g_dark_green + norm_val_for_bin * (base_g_light_green - base_g_dark_green)
        b_val = base_b_dark_green + norm_val_for_bin * (base_b_light_green - base_b_dark_green)
        push!(final_colormap, RGBA(r_val, g_val, b_val, 1.0))
        push!(amenity_greens_sub_colormap, RGBA(r_val, g_val, b_val, 1.0))
    end
    offset_after_public = offset_after_house + N_amenity_bins

    # Colore per il CBD
    id_cbd = offset_after_public + 1
    push!(final_colormap, RGBA(0.2, 0.2, 0.2, 1.0)) 
    total_colormap_entries = length(final_colormap)
    # --- FINE BLOCCO MANCANTE ---
    # Crea la matrice degli ID dei colori
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

    # --- Creazione Plot (Logica di Layout Modificata) ---
    
    # 1. Crea una figura vuota (un po' più larga per la nuova colorbar)
    fig = Figure(resolution = (1000, 900)) 

    # 2. Crea gli assi per la mappa nella PRIMA RIGA, SECONDA COLONNA
    ax = Axis(fig[1, 1], 
              aspect = DataAspect()
    )
    
    # 3. Disegna la heatmap sugli assi
    abmplot!(ax, model;
        agent_size = 0,
        heatarray = _ -> patch_ids_for_colormap,
        heatkwargs = ( colormap = final_colormap, colorrange = (1, length(final_colormap)) ),
        add_colorbar = false
    )

    # 4. Aggiungi il bordo dell'inner city
    r_start, r_end, c_start, c_end = get_inner_city_bounds(model)
    x_coords = [c_start - 0.5, c_end + 0.5, c_end + 0.5, c_start - 0.5, c_start - 0.5]
    y_coords = [r_start - 0.5, r_start - 0.5, r_end + 0.5, r_end + 0.5, r_start - 0.5]
    lines!(ax, x_coords, y_coords, color = :black, linewidth = 2, linestyle = :dash)

    # 5. Aggiungi la Colorbar ORIZZONTALE per l'età, SOTTO il plot
    #    (ora nella SECONDA RIGA, SECONDA COLONNA)
    Colorbar(fig[2, 1], 
        colormap = house_age_sub_colormap, 
        limits = (min_age_plot, max_age_plot), 
        label = "House Age",
        vertical = false,
        height = 30,
        labelsize = 30,
        ticklabelsize = 18,
        flipaxis = false
    )

    colsize!(fig.layout, 1, Aspect(1, 1.0))

    rowgap!(fig.layout, 1, 10) # Spazio tra la mappa e la colorbar orizzontale

    display(fig)
end


function plot_potential_rent_heatmap(model::ABM)
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

     # --- Creazione Plot ---
    # 1. Crea una figura vuota
    fig = Figure(resolution = (1000, 900))

    # 2. Crea gli assi per la mappa
    ax = Axis(fig[1, 1],  aspect = DataAspect())

    # 3. Disegna la heatmap sugli assi
    abmplot!(ax, model;
        agent_size = 0,
        heatarray = _ -> patch_ids_for_colormap,
        heatkwargs = ( colormap = final_colormap, colorrange = (1, length(final_colormap)) ),
        add_colorbar = false # <-- Disattiva la colorbar di default
    )

    # 4. Aggiungi il bordo dell'inner city
    r_start, r_end, c_start, c_end = get_inner_city_bounds(model)
    x_coords = [c_start - 0.5, c_end + 0.5, c_end + 0.5, c_start - 0.5, c_start - 0.5]
    y_coords = [r_start - 0.5, r_start - 0.5, r_end + 0.5, r_end + 0.5, r_start - 0.5]
    lines!(ax, x_coords, y_coords, color = :black, linewidth = 2, linestyle = :dash)

    # 5. Aggiungi la Colorbar personalizzata sotto
    Colorbar(fig[2, 1],
        colormap = potential_rent_sub_colormap, 
        limits = (min_pot_rent_plot, max_pot_rent_plot), 
        label = "Potential Rent",
        vertical = false,
        height = 30,
        labelsize = 30,
        ticklabelsize = 18,
        flipaxis = false  
    )

    colsize!(fig.layout, 1, Aspect(1, 1.0))

    rowgap!(fig.layout, 1, 10) # Spazio tra la mappa e la colorbar orizzontale

    display(fig)
end


function plot_rent_heatmap(model::ABM)
    # Recupera i limiti fissi
    min_rent_plot, max_rent_plot = model.plot_rent_range
    min_amenity_plot, max_amenity_plot = model.plot_amenity_range

    # --- Definizione Colormap ---
    N_rent_bins = 20
    N_amenity_bins = 20
    final_colormap = RGBA{Float64}[]
    id_empty = 1
    push!(final_colormap, RGBA(1.0, 1.0, 1.0, 1.0)) # :empty

    # Colormap per l'affitto
    base_r_low = 0.7; base_g_low = 0.7; base_b_low = 1.0
    base_r_high = 1.0; base_g_high = 0.4; base_b_high = 0.4
    house_rent_sub_colormap = RGBA{Float64}[]
    for i in 0:(N_rent_bins-1)
        norm_val = (N_rent_bins == 1) ? 0.5 : i / (N_rent_bins - 1)
        r = base_r_low + norm_val * (base_r_high - base_r_low)
        g = base_g_low + norm_val * (base_g_high - base_g_low)
        b = base_b_low + norm_val * (base_b_high - base_b_low)
        push!(final_colormap, RGBA(r,g,b,1.0)); push!(house_rent_sub_colormap, RGBA(r,g,b,1.0))
    end
    offset_after_house = id_empty + N_rent_bins

    # Colormap per le amenità
    base_r_dark_g = 0.1; base_g_dark_g = 0.3; base_b_dark_g = 0.1
    base_r_light_g = 0.6; base_g_light_g = 1.0; base_b_light_g = 0.6
    amenity_greens_sub_colormap = RGBA{Float64}[] 
    for i in 0:(N_amenity_bins-1)
        norm_val = (N_amenity_bins == 1) ? 0.5 : i / (N_amenity_bins - 1)
        r = base_r_dark_g + norm_val * (base_r_light_g - base_r_dark_g)
        g = base_g_dark_g + norm_val * (base_g_light_g - base_g_dark_g)
        b = base_b_dark_g + norm_val * (base_b_light_g - base_b_dark_g)
        push!(final_colormap, RGBA(r,g,b,1.0)); push!(amenity_greens_sub_colormap, RGBA(r,g,b,1.0))
    end
    offset_after_public = offset_after_house + N_amenity_bins
    
    id_cbd = offset_after_public + 1
    push!(final_colormap, RGBA(0.2, 0.2, 0.2, 1.0))

    # --- Creazione Matrice ID Colori ---
    patch_ids_for_colormap = zeros(Int, size(model.patch_type))
    range_rent_plot = max_rent_plot - min_rent_plot
    range_amenity_plot = max_amenity_plot - min_amenity_plot

    for idx in CartesianIndices(model.patch_type)
        patch_sym = model.patch_type[idx]
        if patch_sym == :house
            rent_val = model.capitalized_rent[idx]
            norm_rent = !isnan(rent_val) && range_rent_plot > 1e-9 ? (rent_val - min_rent_plot) / range_rent_plot : 0.0
            bin_idx = floor(Int, clamp(norm_rent, 0.0, 1.0) * (N_rent_bins - 1e-9))
            patch_ids_for_colormap[idx] = id_empty + 1 + bin_idx
        elseif patch_sym == :public
            amenity_val = model.amenity_level[idx]
            norm_amenity = !isnan(amenity_val) && range_amenity_plot > 1e-9 ? (amenity_val - min_amenity_plot) / range_amenity_plot : 0.0
            bin_idx = floor(Int, clamp(norm_amenity, 0.0, 1.0) * (N_amenity_bins - 1e-9))
            # --- CORREZIONE QUI ---
            patch_ids_for_colormap[idx] = offset_after_house + 1 + bin_idx
        elseif patch_sym == :cbd
            patch_ids_for_colormap[idx] = id_cbd
        else 
            patch_ids_for_colormap[idx] = id_empty
        end
    end
# --- Creazione Plot (Logica Modificata) ---
    
    # 1. Crea una figura vuota
    fig = Figure(resolution = (1000, 900))

    # 2. Crea gli assi per la mappa nella prima riga
    ax = Axis(fig[1, 1], 
              aspect = DataAspect() # Mantiene le proporzioni corrette
    )
    
    # 3. Disegna la heatmap sugli assi appena creati
    abmplot!(ax, model;
        agent_size = 0,
        heatarray = _ -> patch_ids_for_colormap,
        heatkwargs = ( colormap = final_colormap, colorrange = (1, length(final_colormap)) ),
        add_colorbar = false
    )

    # 4. Aggiungi il bordo dell'inner city
    r_start, r_end, c_start, c_end = get_inner_city_bounds(model)
    x_coords = [c_start - 0.5, c_end + 0.5, c_end + 0.5, c_start - 0.5, c_start - 0.5]
    y_coords = [r_start - 0.5, r_start - 0.5, r_end + 0.5, r_end + 0.5, r_start - 0.5]
    lines!(ax, x_coords, y_coords, color = :black, linewidth = 2, linestyle = :dash)

    # 3. Aggiungi la Colorbar VERTICALE per le amenità, a DESTRA (1, 2)
    Colorbar(fig[1, 2], 
        colormap = amenity_greens_sub_colormap, 
        limits = (min_amenity_plot, max_amenity_plot),
        label = "Amenity Level",
        width = 30, labelsize = 30, ticklabelsize = 18
    )

    # 5. Aggiungi la Colorbar SOTTO, specificando l'altezza e l'orientamento
    Colorbar(fig[2, 1], # Posizione: Seconda riga, prima colonna
        colormap = house_rent_sub_colormap, 
        limits = (min_rent_plot, max_rent_plot), 
        label = "Capitalized Rent",
        vertical = false,      # Rendila orizzontale
        height = 30,           # Aumenta lo spessore
        labelsize = 30,
        ticklabelsize = 18,
        flipaxis = false       # Assicura che i valori vadano da sx a dx
    )
    
    colsize!(fig.layout, 1, Aspect(1, 1.0))

    # 5. Aggiusta il layout
    colgap!(fig.layout, 1, 20) # Spazio tra la mappa e la colorbar verticale
    rowgap!(fig.layout, 1, 10) # Spazio tra la mappa e la colorbar orizzontale

    display(fig)
end


# --- Creazione Plot (Layout 2x2 Definitivo) ---
#= fig = Figure(resolution = (1100, 900))

    # 1. Crea gli assi per la mappa in alto a sinistra (1, 1)
    ax = Axis(fig[1, 1], 
              title = "Model Layout: House Age & Amenities",
              aspect = DataAspect())
    
    abmplot!(ax, model;
        agent_size = 0,
        heatarray = _ -> patch_ids_for_colormap,
        heatkwargs = ( colormap = final_colormap, colorrange = (1, length(final_colormap)) ),
        add_colorbar = false
    )

    # 2. Aggiungi il bordo dell'inner city
    r_start, r_end, c_start, c_end = get_inner_city_bounds(model)
    x_coords = [c_start - 0.5, c_end + 0.5, c_end + 0.5, c_start - 0.5, c_start - 0.5]
    y_coords = [r_start - 0.5, r_start - 0.5, r_end + 0.5, r_end + 0.5, r_start - 0.5]
    lines!(ax, x_coords, y_coords, color = :black, linewidth = 2, linestyle = :dash)

    # 3. Aggiungi la Colorbar VERTICALE per le amenità, a DESTRA (1, 2)
    Colorbar(fig[1, 2], 
        colormap = amenity_greens_sub_colormap, 
        limits = (min_amenity_plot, max_amenity_plot),
        label = "Amenity Level",
        width = 25, labelsize = 22, ticklabelsize = 18
    )
    
    # 4. Aggiungi la Colorbar ORIZZONTALE per l'età, SOTTO (2, 1)
    Colorbar(fig[2, 1], 
        colormap = house_age_sub_colormap, 
        limits = (min_age_plot, max_age_plot), 
        label = "House Age",
        vertical = false,
        height = 30,
        labelsize = 22,
        ticklabelsize = 18,
        flipaxis = true 
    )
    
    # 5. Aggiusta il layout
    colgap!(fig.layout, 1, 10) # Spazio tra la mappa e la colorbar verticale
    rowgap!(fig.layout, 1, 10) # Spazio tra la mappa e la colorbar orizzontale

    display(fig)=#