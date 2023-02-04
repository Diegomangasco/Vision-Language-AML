import os
import logging
import torch
from parse_args import parse_arguments
from load_data import build_splits_baseline, build_splits_domain_disentangle, build_splits_clip_disentangle
from experiments.baseline import BaselineExperiment
from experiments.domain_disentangle import DomainDisentangleExperiment
from experiments.clip_disentangle import CLIPDisentangleExperiment


def setup_experiment(opt):
    
    if opt['experiment'] == 'baseline':
        experiment = BaselineExperiment(opt)
        train_loader, validation_loader, test_loader = build_splits_baseline(opt)
        return experiment, train_loader, validation_loader, test_loader      
    elif opt['experiment'] == 'domain_disentangle':
        experiment = DomainDisentangleExperiment(opt)
        train_loader, validation_loader, test_loader = build_splits_domain_disentangle(opt)
        return experiment, train_loader, validation_loader, test_loader
    elif opt['experiment'] == 'clip_disentangle':
        experiment = CLIPDisentangleExperiment(opt)
        if opt['clip_pretrained'] == 'False':
            train_loader, validation_loader, test_loader, train_clip_loader = build_splits_clip_disentangle(opt)
            return experiment, train_loader, validation_loader, test_loader, train_clip_loader
        else:
            train_loader, validation_loader, test_loader = build_splits_clip_disentangle(opt)
            return experiment, train_loader, validation_loader, test_loader
            
    else:
        raise ValueError('Experiment not yet supported.')
    

def main(opt):
    if opt['experiment'] == 'baseline':
        experiment, train_loader, validation_loader, test_loader = setup_experiment(opt)
    elif opt['experiment'] == 'domain_disentangle':
        experiment, train_loader, validation_loader, test_loader = setup_experiment(opt)
    elif opt['experiment'] == 'clip_disentangle':
        if opt['clip_pretrained'] == 'False':
            experiment, train_loader, validation_loader, test_loader, train_clip_loader = setup_experiment(opt)
        else:
            experiment, train_loader, validation_loader, test_loader = setup_experiment(opt)

    if not opt['test']: # Skip training if '--test' flag is set
        iteration = 0
        best_accuracy = 0
        total_train_loss = 0

        # Restore last checkpoint
        if os.path.exists(f'{opt["output_path"]}/last_checkpoint.pth'):
            iteration, best_accuracy, total_train_loss = experiment.load_checkpoint(f'{opt["output_path"]}/last_checkpoint.pth')
        else:
            logging.info(opt)

        if opt['experiment'] == 'baseline':
            # Train loop
            while iteration < opt['max_iterations']:
                for data in train_loader:

                    total_train_loss += experiment.train_iteration(data)

                    if iteration % opt['print_every'] == 0:
                        print(f'[TRAIN - {iteration}] Loss: {total_train_loss / (iteration + 1)}')
                    
                    if iteration % opt['validate_every'] == 0:
                        # Run validation
                        val_accuracy, val_loss = experiment.validate(validation_loader)
                        print(f'[VAL - {iteration}] Loss: {val_loss} | Accuracy: {(100 * val_accuracy):.2f}')
                        if val_accuracy > best_accuracy:
                            best_accuracy = val_accuracy
                            experiment.save_checkpoint(f'{opt["output_path"]}/best_checkpoint.pth', iteration, best_accuracy, total_train_loss)
                        experiment.save_checkpoint(f'{opt["output_path"]}/last_checkpoint.pth', iteration, best_accuracy, total_train_loss)

                    iteration += 1
                    if iteration > opt['max_iterations']:
                        break
            # Test
            experiment.load_checkpoint(f'{opt["output_path"]}/best_checkpoint.pth')
            test_accuracy, _ = experiment.validate(test_loader)
            logging.info(f'[TEST] Accuracy: {(100 * test_accuracy):.2f}')
            print(f'[TEST] Accuracy: {(100 * test_accuracy):.2f}')

        elif opt['experiment'] == 'domain_disentangle':
            print('Training')
            # Define scheduler
            # A scheduler dynamically changes learning rate
            # The most common schedule is the step(-down), which multiplies learning rate by gamma every STEP_SIZE epochs
            scheduler = torch.optim.lr_scheduler.StepLR(experiment.optimizer, step_size=4, gamma=0.5)
            best_accuracy = 0
            while iteration < opt['max_iterations']:
                logging.info(f'Learning rate {scheduler.get_lr()} at iteration {iteration}')
                for data in train_loader:

                    total_train_loss += experiment.train_iteration(data, train=True)

                    if iteration % opt['print_every'] == 0:
                        print(f'[TRAIN - {iteration}] Loss: {total_train_loss / (iteration + 1)}')
                    
                    if iteration % opt['validate_every'] == 0:
                        # Run validation
                        print("Run validation")
                        val_accuracy, val_loss = experiment.validate(validation_loader, train=False)
                        print(f'[VAL - {iteration}] Loss: {val_loss} | Accuracy: {(100 * val_accuracy):.2f}')
                        if val_accuracy > best_accuracy:
                            print("Saving model...")
                            best_accuracy = val_accuracy
                            experiment.save_checkpoint(f'{opt["output_path"]}/best_checkpoint.pth', iteration, best_accuracy, total_train_loss)
                        experiment.save_checkpoint(f'{opt["output_path"]}/last_checkpoint.pth', iteration, best_accuracy, total_train_loss)

                    iteration += 1
                    if iteration > opt['max_iterations']:
                        break

                scheduler.step()

            # Test
            print("Testing")
            experiment.load_checkpoint(f'{opt["output_path"]}/best_checkpoint.pth')
            test_accuracy, _ = experiment.validate(test_loader, train=False)
            logging.info(f'[TEST] Accuracy: {(100 * test_accuracy):.2f}')
            print(f'[TEST] Accuracy: {(100 * test_accuracy):.2f}')
        
        elif opt['experiment'] == 'clip_disentangle':    
            
            if opt['clip_pretrained'] == 'False':
                print('Clip training')
                while iteration < opt['clip_epochs']:
                    for batch in train_clip_loader:
                        experiment.train_iteration_clip(batch)

                        if iteration % opt['print_every'] == 0:
                            print(f'[CLIP TRAIN - {iteration}]')

                        iteration += 1
                        if iteration > opt['clip_epochs']:
                            break

                experiment.freeze_clip()

            print('Training')
            # Define scheduler
            # A scheduler dynamically changes learning rate
            # The most common schedule is the step(-down), which multiplies learning rate by gamma every STEP_SIZE epochs
            scheduler = torch.optim.lr_scheduler.StepLR(experiment.optimizer, step_size=4, gamma=0.5)
            # Train loops 
            best_accuracy = 0
            iteration = 0
            while iteration < opt['max_iterations']:
                logging.info(f'Learning rate {scheduler.get_lr()} at iteration {iteration}')
                for data in train_loader: # Data is (path, descriptions array)
                    
                    total_train_loss += experiment.train_iteration(data, train=True)

                    if iteration % opt['print_every'] == 0:
                        print(f'[TRAIN - {iteration}] Loss: {total_train_loss / (iteration + 1)}')
                    
                    
                    if iteration % opt['validate_every'] == 0:
                        # Run validation
                        print("Run validation")
                        val_accuracy, val_loss = experiment.validate(validation_loader, train=False)
                        print(f'[VAL - {iteration}] Loss: {val_loss} | Accuracy: {(100 * val_accuracy):.2f}')
                        if val_accuracy > best_accuracy:
                            print("Saving model...")
                            best_accuracy = val_accuracy
                            experiment.save_checkpoint(f'{opt["output_path"]}/best_checkpoint.pth', iteration, best_accuracy, total_train_loss)
                        experiment.save_checkpoint(f'{opt["output_path"]}/last_checkpoint.pth', iteration, best_accuracy, total_train_loss)
                    
                    iteration += 1
                    if iteration > opt['max_iterations']:
                        break

                scheduler.step()

            # Test
            print("Testing")
            experiment.load_checkpoint(f'{opt["output_path"]}/best_checkpoint.pth')
            test_accuracy, _ = experiment.validate(test_loader, train=False)
            logging.info(f'[TEST] Accuracy: {(100 * test_accuracy):.2f}')
            print(f'[TEST] Accuracy: {(100 * test_accuracy):.2f}')
    else:
        # Only testing
        print("Testing")
        experiment.load_checkpoint(f'{opt["output_path"]}/best_checkpoint.pth')
        test_accuracy, _ = experiment.validate(test_loader, train=False)
        logging.info(f'[TEST] Accuracy: {(100 * test_accuracy):.2f}')
        print(f'[TEST] Accuracy: {(100 * test_accuracy):.2f}')
            

if __name__ == '__main__':
    
    logging.getLogger().setLevel(logging.INFO)

    opt = parse_arguments()

    # Setup output directories
    os.makedirs(opt['output_path'], exist_ok=True)
   
    # Setup logger
    logging.basicConfig(filename=f'{opt["output_path"]}/log.txt', format='%(message)s', level=logging.INFO, filemode='a')

    main(opt)
