sigma = 1.0
popsize = 50
optimizer = cma.CMAEvolutionStrategy(input_hidden, sigma, {'popsize': popsize})
for i in range(0,3):
    trial_encodings = optimizer.ask(popsize)
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }
    hiddens_array = np.expand_dims(np.array(trial_encodings), axis=1)
    hiddens_json = {"hiddens": hiddens_array.tolist(),
                    "mask": [[True] for i in range(hiddens_array.shape[0])]}

    response = requests.post(decode_url, headers=headers, json=hiddens_json)
    smiles_list = list(dict.fromkeys(response.json()['generated']))
    smiles_list = get_valid_smiles(smiles_list)
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }
    data = {
        "sequences": smiles_list,
    }
    response = requests.post(hidden_url, headers=headers, json=data)
    hiddens_array = np.squeeze(np.array(response.json()["hiddens"]))
    sim_list = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = mfpgen.GetFingerprint(mol)
            tan_sim = TanimotoSimilarity(target_fp, fp)
            sim_list.append(tan_sim)
    print(max(sim_list))
    optimizer.tell(np.squeeze(np.array(hiddens_array)), sim_list)