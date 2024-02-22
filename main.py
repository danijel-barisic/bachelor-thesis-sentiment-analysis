import numpy as np

from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import sys

SAMPLES = 450
TENTH_OF_SAMP = SAMPLES // 10

PROG_BAR_WIDTH = 20
MUTATION_PROB = 0.1
MUTATION_MAG = 0.1
ELITE_DIVISOR = 10
POPULATION_SIZE = 10

decay_factor = 1.0
curr_gen = 0

# Arithmetic recombination/crossover
def crossover(parent1, parent2):
    # calculating the mean of parents' weights:
    alpha = np.random.rand()
    child = [(alpha * x + (1 - alpha) * y) for x, y in zip(parent1, parent2)]
    return child


def mutate(params_vec, prob=0.01, magnitude=0.1):
    # prob and magnitude are floats in range [0,1] #, magnitude=0.01 optional
    # prob is likelihood of change for individual weight params
    num_of_params_to_mutate = int(len(params_vec) * prob)

    indexes_of_params_to_mutate = list(np.random.choice(len(params_vec), size=num_of_params_to_mutate))
    for i in indexes_of_params_to_mutate:
        params_vec[i] = np.random.normal(loc=params_vec[i], scale=magnitude)

    params_vec_arr = np.asarray(params_vec)
    np.clip(params_vec_arr, -1, 1, out=params_vec_arr)


def select_parents(fit_scores, fit_sum):
    curr_sum = (fit_sum - 0) * np.random.rand() + 0
    for i, fit_score_tuple in enumerate(fit_scores):
        curr_sum += fit_score_tuple[0]
        if curr_sum >= fit_sum:
            return fit_scores[i][1], fit_scores[(i + len(fit_scores) // 2) % len(fit_scores)][1]  # opposite parents


def fitness(weights):
    global model
    model.set_weights(weights_to_matrix(model, weights))
    scores = model.evaluate(X_train, y_train, verbose=0)
    accuracy = scores[1]
    return accuracy


def evaluate(population):
    global decay_factor
    max_score = 0
    print("\nEvaluating generation...", end="")

    evaluated_fitness_tuples = []

    for params_vec in population:
        model.set_weights(weights_to_matrix(model, params_vec))
        scores = model.evaluate(X_test, y_test, verbose=0)
        evaluated_fitness_tuples.append((scores[1], params_vec))

        max_score = max(scores[1], max_score)
    print("\nMax accuracy for this generation: %.2f%%" % (max_score * 100))

    # control point, update decay_factor every 5 generations
    if curr_gen % 5 == 0:
        # square root good on the lows, e.g. 50% accuracy results in 0.25 decay. 90% accuracy is 0.05 decay
        decay_factor = 1 / (10 * max_score + 1) - 0.09

    return evaluated_fitness_tuples


def print_train_bar(progress):
    sys.stdout.flush()
    sys.stdout.write("\rTraining units: [")
    sys.stdout.write("=" * progress + " " * (PROG_BAR_WIDTH - progress))
    sys.stdout.flush()
    sys.stdout.write("]" + " " + "%.2f%%" % ((progress / PROG_BAR_WIDTH) * 100))


def takeFirst(elem):
    return elem[0]


def es(population_size=10, gen_num=100):
    global model
    global curr_gen
    print("\nGenerating the initial population...")
    init_params = model_weights_to_vector(model)
    population = [init_params for i in range(0, population_size)]
    elites = []

    last_gen = False
    curr_gen = 0
    elites_num = 0
    while not last_gen:
        print("\nCurrent generation: {}".format(curr_gen + 1))

        # fitness calculation, on unit weights in population
        print("Calculating fitness...")
        fit_scores_population_tuple = list(map(lambda x: (fitness(x), x), elites + population))
        fit_scores_population_tuple.sort(reverse=True, key=takeFirst)

        is_first_gen = (curr_gen == 0)
        if is_first_gen:
            elites_num = 0
        else:
            elites_num = len(population) // ELITE_DIVISOR
            if elites_num < 1:
                elites_num = 1

        population = make_new_children(fit_scores_population_tuple, elites_num)


        evaluated_fitness_tuples = evaluate(elites + population)
        evaluated_fitness_tuples.sort(reverse=True, key=takeFirst)
        elites_num = len(population) // ELITE_DIVISOR
        if elites_num < 1:
            elites_num = 1
        elites = list(map(lambda t: t[1], evaluated_fitness_tuples[:elites_num]))

        curr_gen += 1
        last_gen = (curr_gen >= gen_num) and gen_num != 0


def make_new_children(fit_scores_population_tuple, elites_num):
    global decay_factor
    children = []

    for i in range(len(fit_scores_population_tuple) - elites_num):
        # select parents
        p1, p2 = select_parents(fit_scores=fit_scores_population_tuple,
                                fit_sum=sum(map(lambda x: x[0], fit_scores_population_tuple)))

        # crossover
        child = crossover(p1, p2)

        # mutate
        mutate(child, prob=MUTATION_PROB * decay_factor, magnitude=MUTATION_MAG * decay_factor)

        # off spring
        children.append(child)

        progress = (len(children) / len(fit_scores_population_tuple)) * PROG_BAR_WIDTH
        print_train_bar(int(progress))
    print_train_bar(PROG_BAR_WIDTH)
    return children


def model_weights_to_vector(model):
    weights_vector = []

    for layer in model.layers:
        if layer.trainable:
            layer_weights = layer.get_weights()
            for l_weights_arr in layer_weights:
                vector = np.reshape(l_weights_arr, newshape=l_weights_arr.size)
                weights_vector.extend(vector)

    return weights_vector


def weights_to_matrix(model, weights_vector):
    weights_matrix = []

    start = 0
    for layer_idx, layer in enumerate(model.layers):
        layer_weights = layer.get_weights()
        if layer.trainable:
            for l_weights in layer_weights:
                layer_weights_shape = l_weights.shape
                layer_weights_size = l_weights.size

                layer_weights_vector = weights_vector[start:start + layer_weights_size]
                layer_weights_matrix = np.reshape(layer_weights_vector, newshape=layer_weights_shape)
                weights_matrix.append(layer_weights_matrix)

                start = start + layer_weights_size
        else:
            for l_weights in layer_weights:
                weights_matrix.append(l_weights)

    return weights_matrix


top_words = 5000

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)[:7 * TENTH_OF_SAMP]
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)[:3 * TENTH_OF_SAMP]
y_train = y_train[:7 * TENTH_OF_SAMP]
y_test = y_test[:3 * TENTH_OF_SAMP]

embedding_vector_length = 32
model = Sequential()

model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

print("\nEvolution algorithm params:")
print("Population size: %d" % POPULATION_SIZE)
print("Mutation probability: %.5f" % MUTATION_PROB)
print("Mutation magnitude: %.5f" % MUTATION_MAG)

es(population_size=POPULATION_SIZE, gen_num=0)