"""
Assignment 2 – Adversarial Image Attack via Hill Climbing

You MUST implement:
    - compute_fitness
    - mutate_seed
    - select_best
    - hill_climb

DO NOT change function signatures.
"""

import json
import os.path

import numpy as np
import matplotlib.pyplot as plt

from typing import List, Tuple
from keras.applications import vgg16
from keras.applications.imagenet_utils import decode_predictions
from keras.utils import array_to_img, load_img, img_to_array
from collections import Counter

from mutation_operators import (
    generate_additive_noise_neighbour,
    generate_local_masking_neighbour,
    channel_specific_perturbation_neighbour,
    generate_lines_neighbour
)

# ============================================================
# 1. FITNESS FUNCTION
# ============================================================

def compute_fitness(
    image_array: np.ndarray,
    model,
    target_label: str
) -> float:
    """
    Compute fitness of an image for hill climbing.

    Fitness definition (LOWER is better):
        - If the model predicts target_label:
              fitness = probability(target_label)
        - Otherwise:
              fitness = -probability(predicted_label)
    """

    preds = model.predict(np.expand_dims(image_array, axis=0), verbose=0)
    labeled_preds = decode_predictions(preds, top=1)[0]

    # Case 1: The model correctly predicted the target label
    if labeled_preds[0][1] == target_label:
        return labeled_preds[0][2]

    # Case 2: The model missclassified
    return - labeled_preds[0][2]


# ============================================================
# 2. MUTATION FUNCTION
# ============================================================

def mutate_seed(
    seed: np.ndarray,
    epsilon: float
) -> List[np.ndarray]:
    """
    Produce ANY NUMBER of mutated neighbours.

    Students may implement ANY mutation strategy:
        - modify 1 pixel
        - modify multiple pixels
        - patch-based mutation
        - channel-based mutation
        - gaussian noise (clipped)
        - etc.

    BUT EVERY neighbour must satisfy the L∞ constraint:

        For all pixels i,j,c:
            |neighbour[i,j,c] - seed[i,j,c]| <= 255 * epsilon

    Requirements:
        ✓ Return a list of neighbours: [neighbour1, neighbour2, ..., neighbourK]
        ✓ K can be ANY size ≥ 1
        ✓ Neighbours must be deep copies of seed
        ✓ Pixel values must remain in [0, 255]
        ✓ Must obey the L∞ bound exactly

    Args:
        seed (np.ndarray): input image
        epsilon (float): allowed perturbation budget

    Returns:
        Tuple[List[np.ndarray], List[str]]: mutated neighbours and their names
    """
    neighbours = []

    #  WARNING: DO NOT CHANGE THE ORDER OF THE MUTATIONS BELOW

    neighbour_additive_noise = generate_additive_noise_neighbour(seed, epsilon)
    neighbours.append(neighbour_additive_noise)

    neighbour_local_masking = generate_local_masking_neighbour(seed, epsilon)
    neighbours.append(neighbour_local_masking)

    neighbour_channel_perturbation = channel_specific_perturbation_neighbour(seed, epsilon)
    neighbours.append(neighbour_channel_perturbation)

    LINE_WIDTH = int(len(seed[0]) * 0.05)
    lines_neighbour = generate_lines_neighbour(seed, epsilon, num_lines=10, width=LINE_WIDTH)
    neighbours.append(lines_neighbour)

    return neighbours


# ============================================================
# 3. SELECT BEST CANDIDATE
# ============================================================

def select_best(
    candidates: List[np.ndarray],
    model,
    target_label: str
) -> Tuple[np.ndarray, float]:
    """
    Evaluate fitness for all candidates and return the one with
    the LOWEST fitness score.

    Args:
        candidates (List[np.ndarray])
        model: classifier
        target_label (str)

    Returns:
        (best_image, best_fitness)
    """
    lowest_score = float('inf')
    best_candidate = candidates[0] # Avoids type warning

    for candidate in candidates:
        fitness = compute_fitness(candidate, model, target_label)
        if fitness < lowest_score:
            lowest_score = fitness
            best_candidate = candidate

    return best_candidate, lowest_score


# ============================================================
# 4. HILL-CLIMBING ALGORITHM
# ============================================================

def L_constraint(seed: np.ndarray, neighbour: np.ndarray, epsilon: float) -> bool:
    diff = np.abs(np.subtract(neighbour, seed))
    return bool(np.all(diff <= 255 * epsilon))

def hill_climb(
    initial_seed: np.ndarray,
    model,
    target_label: str,
    epsilon: float = 0.30,
    iterations: int = 300
) -> Tuple[np.ndarray, float]:
    """
    Main hill-climbing loop.

    Requirements:
        ✓ Start from initial_seed
        ✓ EACH iteration:
              - Generate ANY number of neighbours using mutate_seed()
              - Enforce the SAME L∞ bound relative to initial_seed
              - Add current image to candidates (elitism)
              - Use select_best() to pick the winner
        ✓ Accept new candidate only if fitness improves
        ✓ Stop if:
              - target class is broken confidently, OR
              - no improvement for multiple steps (optional)

    Returns:
        (final_image, final_fitness)
    """

    # Parameters (Not allowed to change function signature, so can be changed here)
    EPSILON_STEP_SIZE = epsilon * 0.05
    NUM_NEIGHBOURS_PER_MUTATION_TYPE = 5
    MAX_ITER_NO_CHANGE = 10
    MUTATION_NAMES = ["Additive Noise", "Local Masking", "Channel Perturbation", "Lines Perturbation"]
    
    # Initialization
    current_fitness = compute_fitness(initial_seed, model, target_label)
    best_fitness = current_fitness
    current_image = initial_seed.copy()
    best_image = initial_seed.copy()
    no_improvement_count = 0
    mutation_history = []

    for iteration in range(iterations):
        # Generate neighbours, zip with mutation names
        neighbours = [n for _ in range(NUM_NEIGHBOURS_PER_MUTATION_TYPE) for n in mutate_seed(current_image, EPSILON_STEP_SIZE)]
        repeated_mutation_names = MUTATION_NAMES * NUM_NEIGHBOURS_PER_MUTATION_TYPE
        
        assert len(neighbours) == len(repeated_mutation_names), "Mismatch between mutation names and generated neighbours"

        neighbours = [(name, img) for name, img in zip(repeated_mutation_names, neighbours)]
        neighbours.append(("Current Image", current_image))

        # Remove i from neighbours if it violates L constraint relative to initial_seed
        valid_neighbours: List[Tuple[str, np.ndarray]] = [(name, img) for name, img in neighbours if L_constraint(initial_seed, img, epsilon)]
        valid_neighbours_images: List[np.ndarray] = [img for _, img in valid_neighbours]

        # Select the best candidate 
        candidate_image, candidate_fitness = select_best(valid_neighbours_images, model, target_label)        

        # Update current image if fitness improved
        if candidate_fitness < current_fitness:
            current_image = candidate_image
            current_fitness = candidate_fitness
            no_improvement_count = 0
            
            # Update global best
            if current_fitness < best_fitness:
                best_image = current_image.copy()
                best_fitness = current_fitness

            # Record mutation used
            for i, (name, img) in enumerate([(name, img) for name, img in valid_neighbours]):  
                if np.array_equal(img, candidate_image):
                    mutation_history.append(name)
                    break
                if i == len(valid_neighbours) - 1:
                    mutation_history.append("If you see this message, something went wrong")

        else:
            no_improvement_count += 1
            
            if (len(valid_neighbours) == 1): 
                mutation_history.append("No valid mutations")
            else: 
                mutation_history.append("No better mutations")
        
        # Stopping conditions
        if no_improvement_count >= MAX_ITER_NO_CHANGE: # can be adjusted
            break
            
        prediction = model.predict(np.expand_dims(current_image, axis=0), verbose=0)
        predicted_class = decode_predictions(prediction, top=1)[0][0][1]
        predicted_confidence = np.max(prediction)

        if predicted_class != target_label and predicted_confidence >= 0.9:
            break

        print(f"Hill Climb Progress: iteration {iteration}/{iterations} | Chosen step: {mutation_history[-1]:>25} | Current Fitness: {current_fitness:.6f} | Best Fitness: {best_fitness:.6f} | No Improvement Count: {no_improvement_count}")

    # Print mutation history in JSON format
    # This was done here because we are not allowed to change function signature
    print("\nMutation counts: \n", Counter(mutation_history))

    # Return best found image and its fitness
    return best_image, best_fitness


# ============================================================
# 5. PROGRAM ENTRY POINT FOR RUNNING A SINGLE ATTACK
# ============================================================

if __name__ == "__main__":
    # Load classifier
    model = vgg16.VGG16(weights="imagenet")

    # Load JSON describing dataset
    with open("data/image_labels.json") as f:
        image_list = json.load(f)

    if not os.path.exists("hc_results"):
        os.makedirs("hc_results")

    EPSILON = 0.2
    ITERATIONS = 200

    # For every image in the list
    for i, item in enumerate(image_list):
        print(f"\n\n\n====== Processing image {i+1}/{len(image_list)} ======")
        filename = item["image"]
        image_path = "images/" + filename
        target_label = item["label"]

        print(f"Loaded image: {image_path}")
        print(f"Target label: {target_label}")

        img = load_img(image_path)
        img_array = img_to_array(img)
        seed = img_array.copy()

        # Get top-5 baseline predictions
        print("\nBaseline predictions (top-5):")
        preds = model.predict(np.expand_dims(seed, axis=0), verbose="0")
        for cl in decode_predictions(preds, top=5)[0]:
            print(f"{cl[1]:20s}  prob={cl[2]:.5f}")

        print("\n")

        # Run hill climbing algortihm and print results
        final_img, final_fitness = hill_climb(
            initial_seed=seed,
            model=model,
            target_label=target_label,
            epsilon=EPSILON,
            iterations=ITERATIONS
        )

        print("\nFinal fitness:", final_fitness)
        final_preds = model.predict(np.expand_dims(final_img, axis=0), verbose="0")

        print("\nFinal predictions:")
        for cl in decode_predictions(final_preds, top=5)[0]:
            print(cl)

        # Save results
        original_save_path = f"hc_results/{filename}_original.png"
        array_to_img(img_array).save(original_save_path)

        adversarial_save_path = f"hc_results/{filename}_adversarial.png"
        array_to_img(final_img).save(adversarial_save_path)

        # Calculate metrics 
        avg_of_changed_pixels = np.mean(np.abs(final_img - seed)) / 255
        max_pixel_change = np.max(np.abs(final_img - seed)) / 255
        num_changed_pixels = np.count_nonzero(np.any(final_img != seed, axis=-1)) / final_img.shape[0] / final_img.shape[1]

        # Store results of the hill climber in JSON format
        report_details = {
            "original_image": original_save_path,
            "adversarial_image": adversarial_save_path,
            "target_label": target_label,
            "epsilon": float(EPSILON),
            "iterations": int(ITERATIONS),
            "original_prediction": { 
                "label": str(decode_predictions(preds, top=1)[0][0][1]),
                "score": float(decode_predictions(preds, top=1)[0][0][2])
            },
            "final_prediction": {
                "label": str(decode_predictions(final_preds, top=1)[0][0][1]),
                "score": float(decode_predictions(final_preds, top=1)[0][0][2]),
                "average_of_pixel_changes": float(avg_of_changed_pixels),
                "maximum_of_pixel_changes": float(max_pixel_change),
                "number_of_changed_pixels": float(num_changed_pixels)
            },
            "final_fitness": float(final_fitness),
            "success": bool(decode_predictions(final_preds, top=1)[0][0][1] != target_label),
            "mutation history": "See terminal output; cannot change function signature"
        }

        report_path = f"hc_results/{filename}_report.json"
        with open(report_path, "w") as f:
            json.dump(report_details, f, indent=4)