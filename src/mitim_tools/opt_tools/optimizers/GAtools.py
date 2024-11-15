import deap
import random
import array
import copy
import datetime
import torch
import numpy as np
import pandas as pd
import dill as pickle
import multiprocessing_on_dill as multiprocessing
from deap import base, benchmarks, tools, algorithms, creator, cma
import deap.benchmarks.tools as bt
from mitim_tools.misc_tools import IOtools, MATHtools
from mitim_tools.opt_tools.OPTtools import summarizeSituation
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed




def findOptima(fun, writeTrajectory=False, **kwargs):
    print("- GA optimization techniques used to maximize acquisition")

    random.seed(fun.seed)

    # Parameters Optimization

    numCases = 1  # Different GAs to launch (random selection of GA params)
    parallel_evaluations_inner = (
        -1
    )  # Refers to inside each optimization (-1: Use all CPUs available)

    # Prepare run

    txtConstr = "unconstrained "  #'constrained '
    print(
        f"\t- Initialization of {txtConstr}GA to solve problem with {fun.dimDVs} DVs, {fun.dimOFs} OFs"
    )

    if parallel_evaluations_inner == -1:
        parallel_evaluations_inner = multiprocessing.cpu_count()
        print(f"\t- Running with {parallel_evaluations_inner} inner processors")

    pop_sizes, max_gens, mut_probs, co_probs = randomizeTrials(
        num=numCases, numOFs=fun.dimOFs, numDVs=fun.dimDVs
    )

    xGuesses = fun.xGuesses.cpu().numpy()
    bounds = fun.bounds_mod.cpu().numpy()

    # Peform workflow
    GA = PRF_GA(
        evaluators=fun.evaluators,
        numOFs=fun.dimOFs,
        dim=fun.dimDVs,
        bounds=bounds,
        pop_sizes=pop_sizes,
        max_gens=max_gens,
        mut_probs=mut_probs,
        co_probs=co_probs,
        xGuesses=xGuesses,
        parallel_evaluations_inner=parallel_evaluations_inner,
        seed=fun.seed,
        stepSettings=fun.stepSettings,
        writeTrajectory=writeTrajectory,
    )

    GA.executeGA()

    acq_evaluated = torch.Tensor(GA.acq_evaluated)

    # Select best pareto front of all cases run

    if numCases == 1:
        GA.besttrial = 0
    else:
        GA.calculatePerformance()
        GA.besttrial = np.argmax(GA.hypervols.values[0])
        print(f"\t\t--> Maximum hypervolume found in trial {GA.besttrial+1}")

    # Grab results

    # Pareto Front
    frontsOfInterest = GA.frontsEvolution[GA.besttrial]
    frontsOfInterest_Pareto = np.atleast_2d(
        frontsOfInterest[0]
    )  # True pareto front is first one after sortDOminated

    # If the pareto front contains fewer points than requested, grab from last population
    membersOfInterest = copy.deepcopy(frontsOfInterest_Pareto)
    members = np.atleast_2d(GA.membersEvolution[GA.besttrial])
    cont = 0
    while membersOfInterest.shape[0] < fun.number_optimized_points:
        membersOfInterest = np.append(
            membersOfInterest, np.atleast_2d(members[cont]), axis=0
        )
        cont += 1

    yObjective = GA.toolboxes[GA.besttrial].evaluate(membersOfInterest)
    HallOfFame = GA.hof[GA.toolboxes[GA.besttrial].experiment_name][
        0
    ]  # Hall of fame contains the best individual ever

    # Pass required values

    x, y, GAOF, HoF = membersOfInterest, yObjective, GA, np.array(HallOfFame)

    # Order results

    order = "highest"

    print(
        f"\t\t- Current population to be passed to optimizer has {x.shape[0]} members"
    )

    if y.shape[0] > 1:
        print(f"\t\t- Ordering Pareto front with first one having the {order} norm")
        x, y = sortGApareto(x, y, order=order)

    # Convert to tensors
    x_opt, y_opt = torch.from_numpy(x).to(fun.stepSettings["dfT"]), torch.from_numpy(
        y
    ).to(fun.stepSettings["dfT"])

    y_opt_residual = summarizeSituation(fun.xGuesses, fun, x_opt)

    # Provide numZ index to track where this solution came from
    numZ = 4
    z_opt = torch.ones(x_opt.shape[0]).to(fun.stepSettings["dfT"]) * numZ

    """
	Notes:	x_opt is unnormalized
			If the problem is MO, this will give a set of points (POINTS,DIM), having the one with highest or lowest norm
			in first place (depending on whether calibration contains -inf)
	"""

    return x_opt, y_opt_residual, z_opt, acq_evaluated


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# GA Class
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class PRF_GA:
    def __init__(
        self,
        evaluators,
        bounds=[0.0, 1.0],
        dim=2,
        numOFs=2,
        weights=None,
        pop_sizes=[50],
        max_gens=[1000],
        mut_probs=[0.2],
        co_probs=[0.7],
        xGuesses=None,
        parallel_evaluations_inner=1,
        seed=0,
        stepSettings={},
        writeTrajectory=False,
    ):
        """
        If residual_function gives y to be maximized (should be in MITIM), then weights must be positive
        """

        if xGuesses is None:
            xGuesses = []

        if weights is None:
            weights = tuple(
                np.ones(numOFs)
            )  # If residual_function gives y to be maximized, then weights must be positive

        self.parallel_evaluations_inner = parallel_evaluations_inner

        random.seed(a=seed)

        self.lower = list(bounds[0, :])
        self.upper = list(bounds[1, :])
        self.dim = dim

        self.runType = "eaMuPlusLambda"

        # ~~~~ Define fitness of individuals

        creator.create("FitnessMin", deap.base.Fitness, weights=weights)
        creator.create(
            "Individual", array.array, typecode="d", fitness=creator.FitnessMin
        )

        # ~~~~ Prepare toolboxes
        toolbox = deap.base.Toolbox()

        # ~~~~ Evaluation toolbox and constraints

        # Input to acq_fun must be (batch,q,dim) or (q,dim)
        self.acq_evaluated = []
        if writeTrajectory:

            def fun_opt(X, v=self.acq_evaluated):
                with torch.no_grad():
                    f = evaluators["acq_function"](
                        torch.Tensor(X).to(stepSettings["dfT"]).unsqueeze(1)
                    )
                v.append(f.max().item())
                return f.unsqueeze(1).cpu().numpy()

        else:

            def fun_opt(X):
                with torch.no_grad():
                    f = evaluators["acq_function"](
                        torch.Tensor(X).to(stepSettings["dfT"]).unsqueeze(1)
                    )
                return f.unsqueeze(1).cpu().numpy()

        toolbox.register("evaluate", fun_opt)

        self.toolboxes = list([toolbox for _ in range(len(pop_sizes))])

        # Define several experiments
        cont = 0
        for pop_size, max_gen, mut_prob, co_prob, toolbox in zip(
            pop_sizes, max_gens, mut_probs, co_probs, self.toolboxes
        ):
            # ~~~~ Define problem configuration and evolutionary process
            toolbox = self.defineParametersEA(
                toolbox,
                pop_size,
                max_gen,
                mut_prob,
                co_prob,
                xGuesses=xGuesses,
                experiment_name=f"Trial {cont}",
            )
            cont += 1

        self.pop_sizes = pop_sizes
        self.max_gens = max_gens
        self.mut_probs = mut_probs
        self.co_probs = co_probs

    def executeGA(self):
        # Run several experiments
        print(
            f"\t- Running a set of {len(self.pop_sizes)} GAs with populations={self.pop_sizes}, generations={self.max_gens}, mutations={self.mut_probs} and cross-overs={self.co_probs}"
        )

        # ~~~~ Run evolution
        self.frontsEvolution = []
        self.info = []
        self.result = {toolbox.experiment_name: [] for toolbox in self.toolboxes}
        self.hof = {toolbox.experiment_name: [] for toolbox in self.toolboxes}
        self.membersEvolution = []

        time1 = datetime.datetime.now()

        i = 0
        for toolbox in self.toolboxes:
            # ----
            print(f"\t\t- Running GA for trial #{i+1}")

            if self.parallel_evaluations_inner > 1:
                print(
                    f"\t\t\t- Parallelizing internal GA with {self.parallel_evaluations_inner} tasks"
                )
                pool = multiprocessing.Pool(processes=self.parallel_evaluations_inner)
                toolbox.register("map", pool.map)

            time1 = datetime.datetime.now()

            fronts, members, result, logbook, hof = run_ea(
                toolbox, runType=self.runType
            )

            print(
                f"\t\t- Run #{i+1} was completed in {IOtools.getTimeDifference(time1)}"
            )
            # ----

            self.frontsEvolution.append(fronts)
            self.info.append(logbook)
            self.membersEvolution.append(members)
            self.result[self.toolboxes[i].experiment_name] = result
            self.hof[self.toolboxes[i].experiment_name] = hof

            i += 1

        print(
            f"\n\t- Optimization took (all runs together) {IOtools.getTimeDifference(time1)}"
        )

    def defineParametersEA(
        self,
        toolbox,
        pop_size,
        max_gen,
        mut_prob,
        co_prob,
        xGuesses=None,
        experiment_name="Trial",
        eta=10.0,
    ):

        if xGuesses is None:
            xGuesses = []

        toolbox.register("attr_float", EAuniform, self.lower, self.upper, self.dim)
        toolbox.register(
            "individual", deap.tools.initIterate, creator.Individual, toolbox.attr_float
        )
        toolbox.register("population", deap.tools.initRepeat, list, toolbox.individual)

        if self.runType == "CMA-ES":
            strategy = cma.Strategy(centroid=[0.0] * self.dim, sigma=0.5)
            toolbox.register("generate", strategy.generate, creator.Individual)
            toolbox.register("update", strategy.update)

        if len(xGuesses) > 0:
            toolbox.register("individual_guess", initIndividual, creator.Individual)
            toolbox.register(
                "population_guess",
                initPopulation,
                list,
                toolbox.individual_guess,
                xGuesses,
            )

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        toolbox.register(
            "mate",
            deap.tools.cxSimulatedBinaryBounded,
            low=self.lower,
            up=self.upper,
            eta=eta,
        )
        toolbox.register(
            "mutate",
            deap.tools.mutPolynomialBounded,
            low=self.lower,
            up=self.upper,
            eta=eta,
            indpb=1.0 / self.dim,
        )
        toolbox.register("select", deap.tools.selNSGA2)

        toolbox.pop_size, toolbox.max_gen = pop_size, max_gen
        toolbox.mut_prob, toolbox.co_prob = mut_prob, co_prob

        toolbox.experiment_name = experiment_name.format(
            toolbox.pop_size, toolbox.max_gen
        )

        return toolbox

    def calculatePerformance(self):
        print("\t\t\t--> Calculating Performance Metrics")
        reference = calculate_reference(self.result, self.toolboxes[0], epsilon=0.1)

        res = pd.DataFrame(self.result)
        # res.reindex_axis([toolbox.experiment_name for toolbox in self.toolboxes], axis=1)
        self.hypervols = res.applymap(lambda pop: bt.hypervolume(pop, reference))

    def pareto_dominance(self, x, y):
        return tools.emo.isDominated(x.fitness.values, y.fitness.values)

    def evaluateIndividual(self, x):
        a_given_individual = self.toolbox.population(n=1)[0]
        for cont, i in enumerate(x):
            a_given_individual[cont] = i

        a_given_individual.fitness.values = self.toolbox.evaluate(a_given_individual)

        return a_given_individual

    def determineDomination(self, a_given_individual, example_pop):
        dominated = [
            ind for ind in example_pop if self.pareto_dominance(a_given_individual, ind)
        ]
        dominators = [
            ind for ind in example_pop if self.pareto_dominance(ind, a_given_individual)
        ]
        others = [
            ind for ind in example_pop if not ind in dominated and not ind in dominators
        ]

        return dominated, dominators, others

    def determinePareto(self, example_pop):
        return tools.sortNondominated(
            example_pop, k=len(example_pop), first_front_only=True
        )[0]

    def save(self, stateFile):
        with open(stateFile, "wb") as handle:
            pickle.dump(self, handle, protocol=4)
        print(f" --> GA state file {stateFile} generated, containing the PRF_GA class")

    def readGA(self, stateFile):
        with open(stateFile, "rb") as f:
            GA = pickle.load(f)

        return GA


def calculate_reference(results, toolbox, epsilon=0.1):
    alldata = []
    for i in results:
        alldata.extend(results[i][0])

    obj_vals = toolbox.evaluate(alldata)
    return np.max(obj_vals, axis=0) + epsilon


def run_ea(toolbox, runType="eaMuPlusLambda"):
    # Store information
    stats1 = deap.tools.Statistics()
    stats1.register("pop", copy.deepcopy)
    stats2 = deap.tools.Statistics(key=lambda ind: ind.fitness.values)
    stats2.register("avg", np.mean)
    stats2.register("std", np.std)
    stats2.register("min", np.min)
    stats2.register("max", np.max)
    mstats = deap.tools.MultiStatistics(fitness=stats2, members=stats1)
    # ------------------------------------------------------------------------

    try:
        pop_guess = toolbox.population_guess()
    except:
        pop_guess = []
    pop = toolbox.population(n=toolbox.pop_size - len(pop_guess))
    if len(pop_guess) > 0:
        pop.extend(pop_guess)

    print(
        f"\t\t\t- From a total of {len(pop)}, {len(pop_guess)} members were guessed & {len(pop)-len(pop_guess)} random"
    )

    # ---------------------------------
    # Running algorithm
    # ---------------------------------

    if runType == "eaMuPlusLambda":
        hof = tools.HallOfFame(1)

        population, logbook = eaMuPlusLambda(
            pop,
            toolbox,
            mu=toolbox.pop_size,  # Number of individuals to select for the next generation
            ngen=toolbox.max_gen,  # Number of generations
            lambda_=toolbox.pop_size
            // 2,  # Number of children to produce at each generation
            cxpb=toolbox.co_prob,  # Probability that an offspring is produced by crossover
            mutpb=toolbox.mut_prob,  # Probability that an offspring is produced by mutation
            stats=mstats,
            halloffame=hof,
            verbose=False,
        )

    else:
        raise Exception("Method not implemented yet in MITIM")

    # ---------------------------------------------------------
    # result is the entire population at final generation
    # ---------------------------------------------------------

    fronts, allpoints = getFinalFronts(population, logbook, toolbox)

    info = {
        "gen": np.array(logbook.chapters["fitness"].select("gen")),
        "avg": np.array(logbook.chapters["fitness"].select("avg")),
        "min": np.array(logbook.chapters["fitness"].select("min")),
        "max": np.array(logbook.chapters["fitness"].select("max")),
    }

    return np.array(fronts), np.array(allpoints), population, info, hof


def getFinalFronts(population, logbook, toolbox, getAll=False):
    """
    population is the entire set of points at the last evaluation

    the output "fronts" contains the best (pareto or direct evaluation) at each generation.
    In the case of not getAll, just the last one

    """

    if getAll:
        print("\t\t\t~~~~ Getting all fronts")
        fronts = []
        allpoints = []
        for i in range(len(logbook.select("pop"))):
            members = logbook.chapters["members"][i]["pop"]
            allpoints.append(members)
            try:
                try:
                    fr = deap.tools.sortLogNondominated(members, len(members))
                except:
                    fr = deap.tools.sortNondominated(members, len(members))
                fronts.append(fr[0])
            except:
                print("~~~~ Problem retrieving all fronts")
                getAll = False
                break

    if not getAll:
        print("\t\t\t~~~~ Getting Pareto frontier at last generation or best case")
        fitness_lastpop = toolbox.evaluate(population)

        if fitness_lastpop.shape[1] == 1:
            fronts = [population[fitness_lastpop.argmax()]]
        else:
            try:
                fronts = deap.tools.sortLogNondominated(population, len(population))
            except:
                fronts = deap.tools.sortNondominated(population, len(population))
            # First front is pareto front

        allpoints = population

    return fronts, allpoints


def initPopulation(pcls, ind_init, guessedPop):
    return pcls(ind_init(c) for c in guessedPop)


def initIndividual(icls, content):
    return icls(content)


def EAuniform(low, up, size=None):
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]


def applyRecurrentNiche(x, y, q=2, Initol=1e-2):
    # x should be bounded between 0 and 1

    print("\t--> Application of recurrent niching")
    print(f"\t\t--> Current population has {x.shape[0]} members")

    if x.shape[0] == 1:
        print("\t\t\t--> Pareto population has only one member, no niching applicable")
        return x, y
    else:
        print(
            "\t\t\t--> Applying default niche to initiallly {0} samples to avoid super-clustering".format(
                x.shape[0]
            )
        )
        x, y = MATHtools.applyNiche(x, y=y, tol=1e-2)
        print(f"\t\t\t--> Current population has now {x.shape[0]} members")

        print(
            "\t\t\t--> Applying DVs recurrent niche method to reduce {0} individuals to (at most) {1} individuals".format(
                x.shape[0], q
            )
        )
        tol = Initol
        while x.shape[0] > q:
            x, y = MATHtools.applyNiche(x, y=y, tol=tol)
            # print('\t\t\t\t--> x has now {0} individuals'.format(x.shape[0]))
            tol = tol * 1.1
        print(f"\t\t--> Current population has now {x.shape[0]} members")

        return x, y


def sortGApareto(x, y, order="highest", orderNorm=1):
    """
    Sorts (x,y) following the metric of norm.
    Because my setup follows a MAXIMIZATION problem, order='highest' will
    put in first place the one with the highest mean
    """

    print("\t\t\t- Applying very tight niching correction...")
    x, y = MATHtools.applyNiche(x, y=y, tol=1e-4)

    print(f"\t\t\t- Sorting by L2-norm ({order} first)")

    if order == "highest":
        dire = -1  # Highest numbers first
    else:
        dire = +1  # Lowest numbers first

    # ---- Metric to order ------
    l2 = dire * np.linalg.norm(y, axis=1, ord=orderNorm)
    # ---------------------------

    xN = MATHtools.orderArray(x, base=l2)
    yN = MATHtools.orderArray(y, base=l2)

    return xN, yN


def randomizeTrials(num=2, numOFs=2, numDVs=2):
    maxPopulationSize = (
        1000  # My tensorial formulation works great at high number, parallel
    )
    minPopulationSize = int(maxPopulationSize / 2)
    generations = 500  # np.min([1000,50*numDVs])

    pop_sizes, max_gens, mut_probs, co_probs = [], [], [], []
    for i in range(num):
        pop_sizes.append(random.randint(int(minPopulationSize), int(maxPopulationSize)))
        max_gens.append(generations)
        mut_probs.append(random.randint(5, 10) / 100.0)
        co_probs.append(random.randint(60, 80) / 100.0)

    return pop_sizes, max_gens, mut_probs, co_probs


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# GA Optimization Information
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def GAessentials(GAoriginal, x_next, train_X, Norm_Xmin, Norm_Xmax):
    if GAoriginal is not None:
        fronts, fronts_y = convertGroup(
            GAoriginal.frontsEvolution[GAoriginal.besttrial][0],
            GAoriginal.toolboxes[GAoriginal.besttrial],
        )
        alls, alls_y = convertGroup(
            GAoriginal.membersEvolution[GAoriginal.besttrial],
            GAoriginal.toolboxes[GAoriginal.besttrial],
        )

        toolbox = GAoriginal.toolboxes[GAoriginal.besttrial]

        # Remove unpickle (possible pool) objects
        try:
            del toolbox.map
        except:
            pass

        fronts = np.atleast_2d(fronts)
        fronts_y = np.atleast_2d(fronts_y)

        # Unnormalize
        fronts_unnormalized = copy.deepcopy(fronts)
        from mitim_tools.opt_tools.SURROGATEtools import denormalizeVar

        for i in range(fronts.shape[1]):
            fronts_unnormalized[:, i] = denormalizeVar(
                torch.from_numpy(fronts[:, i]), Norm_Xmin[i], Norm_Xmax[i]
            )

        GA = {
            "All_x": alls,
            "All_y": alls_y,
            "Paretos_x": fronts,
            "Paretos_x_unnormalized": fronts_unnormalized,
            "Paretos_y": fronts_y,
            "fitness": GAoriginal.info[GAoriginal.besttrial],
            "x": np.array(x_next),
            "y": None,
            "y_res": None,
            "trained": np.array(train_X),
        }

    else:
        GA = {}

    return GA


def convertGroup(frontsEvolution, toolboxes, obtainY=False):
    fronts = []
    frontsx = []
    for gen in range(len(frontsEvolution)):
        if obtainY:
            frontsG = toolboxes.evaluate(frontsEvolution[gen])
        else:
            frontsG = []
        frontsxG = frontsEvolution[gen]
        fronts.append(np.array(frontsG))
        frontsx.append(np.array(frontsxG))

    return np.array(frontsx), np.array(fronts)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Modify original function to accept batches instead of loops. Much faster!
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def print_summary(i, ngen, fitness_best, individuals, each=50):
    if i % each == 0:
        print(
            f'\t\t\t\t* Generation {str(i).rjust(3)}/{ngen} ({str(individuals).rjust(4)} individuals), best acquisition = {f"{fitness_best:.2e}".rjust(9)}',
        )


from deap.algorithms import varOr


def eaMuPlusLambda(
    population,
    toolbox,
    mu,
    lambda_,
    cxpb,
    mutpb,
    ngen,
    stats=None,
    halloffame=None,
    verbose=__debug__,
):
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]

    fitnesses = toolbox.evaluate(invalid_ind)
    for i, ind in enumerate(invalid_ind):
        ind.fitness.values = fitnesses[i]
    print_summary(0, ngen, fitnesses.max(), fitnesses.shape[0])

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Vary the population
        offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

        fitnesses = toolbox.evaluate(invalid_ind)
        for i, ind in enumerate(invalid_ind):
            ind.fitness.values = fitnesses[i]

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Select the next generation population
        population[:] = toolbox.select(population + offspring, mu)

        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        print_summary(gen, ngen, population[0].fitness.values[0], len(population))

    return population, logbook
