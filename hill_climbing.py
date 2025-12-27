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
from scipy.ndimage import gaussian_filter


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

    preds = model.predict(np.expand_dims(image_array, axis=0))
    labeled_preds = decode_predictions(preds, top=1)[0]

    # Case 1: The model correctly predicted the target label
    if labeled_preds[0][1] == target_label:
        return labeled_preds[0][2]

    # Case 2: The model missclassified
    return - labeled_preds[0][2]


# ============================================================
# 2. MUTATION FUNCTION
# ============================================================

def generate_additive_noise_neighbour(seed: np.ndarray, epsilon: float) -> np.ndarray:
    neighbour = seed.copy()
    h, w, c = seed.shape
    limit = 255 * epsilon
    noise = np.random.uniform(-limit, limit, (h, w, c))
    neighbour = np.clip(neighbour + noise, 0, 255)
    return neighbour

def generate_local_masking_neighbour(seed: np.ndarray, epsilon: float) -> np.ndarray:
    neighbour = gaussian_filter(seed, sigma=(1.0, 1.0, 0.0))
    neighbour = neighbour.astype(np.float32)

    limit = 255 * epsilon
    mask = np.abs(neighbour - seed) < limit
    neighbour = np.where(mask, neighbour, seed)
    neighbour = np.clip(neighbour, 0, 255)
    return neighbour

def channel_specific_perturbation_neighbour(seed: np.ndarray, epsilon: float) -> np.ndarray:
    neighbour = seed.copy()
    h, w, c = seed.shape
    limit = 255 * epsilon
    for channel in range(c):
        noise = np.random.uniform(-limit, limit, (h, w))
        neighbour[:, :, channel] = np.clip(neighbour[:, :, channel] + noise, 0, 255)
    return neighbour

def generate_line_stripe_neighbour(seed: np.ndarray, epsilon: float, num_lines: int = 3, width: int = 3) -> np.ndarray:
    neighbour = seed.copy()
    h, w, c = seed.shape
    limit = 255 * epsilon

    X, Y = np.meshgrid(np.arange(w), np.arange(h))

    for _ in range(num_lines):
        slope = np.random.uniform(-5.0, 5.0)
        intercept = np.random.uniform(0, h) - slope * (0.5 * w) 

        # compute distance from line y = slope * x + intercept
        dist = np.abs(Y - (slope * X + intercept))
        mask = dist <= width  # boolean mask where line affects

        # generate random perturbation for masked pixels
        delta = np.random.uniform(-limit, limit, size=(h, w, c))
        for ch in range(c):
            clipped  = np.clip(neighbour[:, :, ch] + delta[:, :, ch], 0, 255)
            neighbour[:, :, ch] = np.where(mask, clipped, neighbour[:, :, ch])

    return neighbour

def L_constraint(seed: np.ndarray, neighbour: np.ndarray, epsilon: float) -> bool:
    diff = np.abs(np.subtract(neighbour, seed))
    return bool(np.all(diff <= 255 * epsilon))

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
        List[np.ndarray]: mutated neighbours
    """

    neighbours = []

    neighbour_additive_noise = generate_additive_noise_neighbour(seed, epsilon)
    if L_constraint(seed, neighbour_additive_noise, epsilon):
        neighbours.append(neighbour_additive_noise)

    neighbour_local_masking = generate_local_masking_neighbour(seed, epsilon)
    if L_constraint(seed, neighbour_local_masking, epsilon):
        neighbours.append(neighbour_local_masking)

    neighbour_channel_perturbation = channel_specific_perturbation_neighbour(seed, epsilon)
    if L_constraint(seed, neighbour_channel_perturbation, epsilon):
        neighbours.append(neighbour_channel_perturbation)

    stripe_width = int(len(seed[0]) * 0.05)
    line_stripe_neighbour = generate_line_stripe_neighbour(seed, epsilon, num_lines=10, width=stripe_width)
    if L_constraint(seed, line_stripe_neighbour, epsilon):
        neighbours.append(line_stripe_neighbour)

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
    current_fitness = compute_fitness(initial_seed, model, target_label)
    best_fitness = current_fitness
    
    current_image = initial_seed.copy()
    best_image = initial_seed.copy()
    
    no_improvement_count = 0

    for iteration in range(iterations):
        # Generate neighbours
        neighbours = mutate_seed(current_image, epsilon)
        neighbours.append(current_image)

        # Remove i from neighbours if it violates L constraint relative to initial_seed
        neighbours = [n for n in neighbours if L_constraint(initial_seed, n, epsilon)]

        # Select the best candidate
        candidate_image, candidate_fitness = select_best(neighbours, model, target_label)

        if candidate_fitness < current_fitness:
            current_image = candidate_image
            current_fitness = candidate_fitness
            no_improvement_count = 0
            
            # Update global best
            if current_fitness < best_fitness:
                best_image = current_image.copy()
                best_fitness = current_fitness
        else:
            no_improvement_count += 1
        
        # Stopping conditions
        if no_improvement_count >= 40: # can be adjusted
            break
            
        prediction = model.predict(np.expand_dims(current_image, axis=0), verbose=0)
        predicted_class = decode_predictions(prediction, top=1)[0][0][1]
        predicted_confidence = np.max(prediction)

        if predicted_class != target_label and predicted_confidence >= 0.9:
            break

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

    EPSILON = 0.30
    ITERATIONS = 50

    # For every image in the list
    for i, item in enumerate(image_list):
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
        preds = model.predict(np.expand_dims(seed, axis=0))
        for cl in decode_predictions(preds, top=5)[0]:
            print(f"{cl[1]:20s}  prob={cl[2]:.5f}")

        # Run hill climbing algortihm and print results
        final_img, final_fitness = hill_climb(
            initial_seed=seed,
            model=model,
            target_label=target_label,
            epsilon=EPSILON,
            iterations=ITERATIONS
        )

        print("\nFinal fitness:", final_fitness)
        final_preds = model.predict(np.expand_dims(final_img, axis=0))

        print("\nFinal predictions:")
        for cl in decode_predictions(final_preds, top=5)[0]:
            print(cl)

        # Save results
        original_save_path = f"hc_results/{filename}_original.png"
        array_to_img(img_array).save(original_save_path)

        adversarial_save_path = f"hc_results/{filename}_adversarial.png"
        array_to_img(final_img).save(adversarial_save_path)

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
                "score": float(decode_predictions(final_preds, top=1)[0][0][2])
            },
            "final_fitness": float(final_fitness),
            "mutation": "CHANGE BASED ON CHOSEN MUTATION"
        }

        report_path = f"hc_results/{filename}_report.json"
        with open(report_path, "w") as f:
            json.dump(report_details, f, indent=4)