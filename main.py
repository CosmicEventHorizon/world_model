from samplers.Sampler import Sampler
from trainers.VAETrainer import VAETrainer
from trainers.VelocityControllerTrainer import VelocityControllerTrainer

if __name__ == "__main__":
    print("Choose from one of the following options below:\n")
    print("****************************************")
    print("1: Collect samples for training")
    print("2: Train the VAE Model")
    print("3: Train the Velocity Controller Model")
    print("****************************************\n")
    answer = int(input("Choice: "))
    match answer:
        case 1:
            Sampler.sample(100000)
        case 2:
            VAETrainer.train(20)
        case 3:
            VelocityControllerTrainer.train(500)
        case _:
            print("Unknown option, exiting...")
            quit()
