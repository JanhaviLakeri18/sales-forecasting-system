def select_best_model(results):

    best_model = min(
        results,
        key=lambda x: x['mae']
    )

    print("\nBest Model Selected")

    print(best_model['name'])

    return best_model