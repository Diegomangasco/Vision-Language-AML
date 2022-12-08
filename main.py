import os
import logging
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
        train_loader_source, train_loader_target, validation_loader_source, validation_loader_target, test_loader = build_splits_domain_disentangle(opt)
        return experiment, train_loader_source, train_loader_target, validation_loader_source, validation_loader_target, test_loader

    elif opt['experiment'] == 'clip_disentangle':
        experiment = CLIPDisentangleExperiment(opt)
        train_loader, validation_loader, test_loader = build_splits_clip_disentangle(opt)

    else:
        raise ValueError('Experiment not yet supported.')
    
    return experiment, train_loader, validation_loader, test_loader

def main(opt):
    if opt['experiment'] == 'baseline':
        experiment, train_loader, validation_loader, test_loader = setup_experiment(opt)
    elif opt['experiment'] == 'domain_disentangle':
        experiment, train_loader_source, train_loader_target, validation_loader_source, validation_loader_target, test_loader = setup_experiment(opt)
    elif opt['experiment'] == 'clip_disentangle':
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
            print('Train loop with source')
            # Train loop with source
            while iteration < opt['max_iterations']:
                for data in train_loader_source:

                    total_train_loss += experiment.train_iteration(data, label=0)

                    if iteration % opt['print_every'] == 0:
                        print(f'[TRAIN - {iteration}] Loss: {total_train_loss / (iteration + 1)}')
                    
                    if iteration % opt['validate_every'] == 0:
                        # Run validation
                        val_accuracy, val_loss = experiment.validate(validation_loader_source, label=0)
                        print(f'[VAL - {iteration}] Loss: {val_loss} | Accuracy: {(100 * val_accuracy):.2f}')
                        if val_accuracy > best_accuracy:
                            experiment.save_checkpoint(f'{opt["output_path"]}/best_checkpoint.pth', iteration, best_accuracy, total_train_loss)
                        experiment.save_checkpoint(f'{opt["output_path"]}/last_checkpoint.pth', iteration, best_accuracy, total_train_loss)

                    iteration += 1
                    if iteration > opt['max_iterations']:
                        break
            iteration = 0
            best_accuracy = 0
            total_train_loss = 0
            print('Train loop with target')
            # Train loop with target
            while iteration < opt['max_iterations']:
                for data in train_loader_target:

                    total_train_loss += experiment.train_iteration(data, label=1)

                    if iteration % opt['print_every'] == 0:
                        print(f'[TRAIN - {iteration}] Loss: {total_train_loss / (iteration + 1)}')
                    
                    if iteration % opt['validate_every'] == 0:
                        # Run validation
                        val_accuracy, val_loss = experiment.validate(validation_loader_target, label=1)
                        print(f'[VAL - {iteration}] Loss: {val_loss} | Accuracy: {(100 * val_accuracy):.2f}')
                        if val_accuracy > best_accuracy:
                            experiment.save_checkpoint(f'{opt["output_path"]}/best_checkpoint.pth', iteration, best_accuracy, total_train_loss)
                        experiment.save_checkpoint(f'{opt["output_path"]}/last_checkpoint.pth', iteration, best_accuracy, total_train_loss)

                    iteration += 1
                    if iteration > opt['max_iterations']:
                        break
            # Test
            experiment.load_checkpoint(f'{opt["output_path"]}/best_checkpoint.pth')
            test_accuracy, _ = experiment.validate(test_loader, label=2)
            logging.info(f'[TEST] Accuracy: {(100 * test_accuracy):.2f}')
            print(f'[TEST] Accuracy: {(100 * test_accuracy):.2f}')

        elif opt['experiment'] == 'clip_disentangle':
            print('Train loop with source')
            # Train loop with source
            while iteration < opt['max_iterations']:
                for data in train_loader_source:

                    total_train_loss += experiment.train_iteration(data, label=0)

                    if iteration % opt['print_every'] == 0:
                        print(f'[TRAIN - {iteration}] Loss: {total_train_loss / (iteration + 1)}')
                    
                    if iteration % opt['validate_every'] == 0:
                        # Run validation
                        val_accuracy, val_loss = experiment.validate(validation_loader_source, label=0)
                        print(f'[VAL - {iteration}] Loss: {val_loss} | Accuracy: {(100 * val_accuracy):.2f}')
                        if val_accuracy > best_accuracy:
                            experiment.save_checkpoint(f'{opt["output_path"]}/best_checkpoint.pth', iteration, best_accuracy, total_train_loss)
                        experiment.save_checkpoint(f'{opt["output_path"]}/last_checkpoint.pth', iteration, best_accuracy, total_train_loss)

                    iteration += 1
                    if iteration > opt['max_iterations']:
                        break
            iteration = 0
            best_accuracy = 0
            total_train_loss = 0
            print('Train loop with target')
            # Train loop with target
            while iteration < opt['max_iterations']:
                for data in train_loader_target:

                    total_train_loss += experiment.train_iteration(data, label=1)

                    if iteration % opt['print_every'] == 0:
                        print(f'[TRAIN - {iteration}] Loss: {total_train_loss / (iteration + 1)}')
                    
                    if iteration % opt['validate_every'] == 0:
                        # Run validation
                        val_accuracy, val_loss = experiment.validate(validation_loader_target, label=1)
                        print(f'[VAL - {iteration}] Loss: {val_loss} | Accuracy: {(100 * val_accuracy):.2f}')
                        if val_accuracy > best_accuracy:
                            experiment.save_checkpoint(f'{opt["output_path"]}/best_checkpoint.pth', iteration, best_accuracy, total_train_loss)
                        experiment.save_checkpoint(f'{opt["output_path"]}/last_checkpoint.pth', iteration, best_accuracy, total_train_loss)

                    iteration += 1
                    if iteration > opt['max_iterations']:
                        break
            # Test
            experiment.load_checkpoint(f'{opt["output_path"]}/best_checkpoint.pth')
            test_accuracy, _ = experiment.validate(test_loader, label=2)
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
