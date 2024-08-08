import argparse
import pickle
import torchvision
import nas
import os
import time


def run(mu_arg, lambda_arg, generations):

    con = nas.population.Container()
    con.add(
        layer_class="CL", kernel_sizes=[2, 3, 5], strides=[1], out_channels=[16, 32, 64]
    )
    con.add(layer_class="PL", layer_subclass="AVG", kernel_sizes=[2], strides=[2])
    con.add(layer_class="PL", layer_subclass="MAX", kernel_sizes=[2], strides=[2])
    con.add(layer_class="FC", out_features=[32, 64])
    con.add(layer_class="ML", layer_subclass="CON")
    con.add(layer_class="ML", layer_subclass="ADD")
    con.add(layer_class="OUTPUT", out_features=[10])

    tpl = nas.population.cnn.Template(
        L_back=2,
        rows=(
            "CL",
            "CL",
            "PL",
            "PL",
            "CL",
            "CL",
            "PL",
            "CL",
            "PL",
            "CL",
            "CL",
            "PL",
            "PL",
            "FC",
            "FC",
        ),
        cols=[7] * 15,
    )

    gc = nas.population.cnn.GraphController(template=tpl, ML_probability=0.1, container=con)

    gen = nas.population.cnn.Generator(graph_controller=gc, container=con)

    gen.calculate_mua = False

    ter = nas.Terminator(max_generations=generations, run_time=30 * 60 * 60)

    nsga2 = nas.opt.mo.NSGA_II(
        generator=gen,
        termination_strategy=ter.generation_termination,
        batch_size=4,
        dataset=torchvision.datasets.MNIST,
        num_epochs=4,
        mu_arg=mu_arg,
        lambda_arg=lambda_arg,
        mutation_rate=0.1,
        training_size=20000,
        test_size=10000,
    )

    data = nsga2.evolve()

    # Create a directory to store the files if it doesn't exist
    directory = "data_files"
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Get the next available run number
    run_number = get_next_run_number(mu_arg, lambda_arg)

    # Create file names based on the arguments and the run number
    file_name = f"run_{run_number}_mu_{mu_arg}_lambda_{lambda_arg}.pkl"
    file_path = os.path.join(directory, file_name)

    # Write some data to the files
    with open(file_path, "wb") as f:
        pickle.dump(data, f)


def get_next_run_number(mu_arg, lambda_arg):
    # Function to get the next available run number based on the arguments
    directory = "data_files"
    if not os.path.exists(directory):
        return 1

    for run_number in range(1, 10000):
        file_exists = os.path.exists(
            os.path.join(
                directory, f"run_{run_number}_mu_{mu_arg}_lambda_{lambda_arg}.pkl"
            )
        )
        if not file_exists:
            return run_number

    exit(0)

def positive_int(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} is not a positive integer")
    return ivalue

def main():
    parser = argparse.ArgumentParser(description="This script implements an evolutionary algorithm using the (mu + lambda) strategy, where mu represents the number of parents selected for reproduction and lambda represents the number of offspring generated per generation.")
    parser.add_argument("--mu", type=positive_int, help="Value for mu (positive integer)")
    parser.add_argument("--lambda", type=positive_int, help="Value for lambda (positive integer)")
    parser.add_argument("--generations", type=positive_int, help="Number of generations (positive integer)")
    args = parser.parse_args()
    run(mu_arg = getattr(args, "mu"), lambda_arg = getattr(args, "lambda"), generations = getattr(args, "generations"))

if __name__ == "__main__":
    main()