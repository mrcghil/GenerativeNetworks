import yaml, os, itertools, copy
import numpy as np

input_path = os.path.abspath('C:\\WORKSPACES\\ZINKY\\GenerativeNetworks\\InputFiles\\scary_colorful_images.yaml')

class Simulation(object):
    attribute_dictionary = {
        'Mood': 'mood',
        'Subject': 'subject',
        'Style': 'style',
        'Seed': 'seed',
        'InitialWeight': 'initial_image_weight',
        'FullString': 'full_string',
        'InitialImage': 'initial_image',
        'ObjectiveImage': 'objective_image',
        'LearningRate': 'learning_rate',
        'MaxIterations': 'max_iterations', 
    }

    def __init__(self, settings: dict):
        self.results_path_base = settings['ResultPath']
        self.subfolder_keyword = settings['Subfolder']
        self.subfolder = ''
        self.save_path = ''
        self.model_name = settings['Model']
        if settings['FullString'] == '' or settings['AssembleString']:
            self.mood = settings['Mood']
            self.subject = settings['Subject']
            self.style = settings['Style']
            self.full_string = self.assemble_fullstring(
                settings['Mood'], 
                settings['Subject'],
                settings['Style'],
            )
        else:
            self.mood, self.subject, self.style = ''
            self.full_string = settings['FullString']
        # Setting initial and final images 
        self.initial_image = settings['InitialImage']
        self.initial_image_weight = settings['InitialImageWeight']
        self.objective_image = settings['ObjectiveImage']
        # Other parameters
        self.seed = int(settings['Seed'])
        self.height = int(settings['Dimensions'][0])
        self.width = int(settings['Dimensions'][1])
        self.learning_rate = settings['LearningRate']
        self.max_iterations = settings['MaxIterations']
        # Verification
        self.is_valid = None
        self.validate_settings()

    @staticmethod
    def define_subfolder(parent_folder:str, cue_name:str = 'Test') -> str:
        i = 1
        added_folder = ''.join([cue_name, str(i).zfill(4)])
        foldername = os.path.join(parent_folder, added_folder)
        while os.path.exists(foldername):
            i += 1
            added_folder = ''.join([cue_name, str(i).zfill(4)])
            foldername = os.path.join(parent_folder, added_folder)
        return added_folder

    @staticmethod
    def assemble_fullstring(_mood:str, _subject:str, _style:str) -> str:
        fullstring = ' '.join([_mood, _subject, 'in', _style])
        return fullstring
    
    def to_dictionary(self):
        attributes_to_save = self.attribute_dictionary.values()
        dict_to_return = {}
        for item in attributes_to_save:
            dict_to_return[item] = getattr(self, item)
        return dict_to_return

    def save_sim_settings(self):
        settings_path = os.path.join(self.save_path, 'SimSettings.yml')
        with open(settings_path, 'w+') as f:
            yaml.dump(self.to_dictionary(), f, sort_keys = False, default_flow_style = False)
            print(f'Simulation.save_sim_settings :: Saved settings {settings_path}')

    def generate_path(self):
        self.subfolder = self.define_subfolder(self.results_path_base, cue_name = self.subfolder_keyword)
        self.save_path = os.path.join(self.results_path_base, self.subfolder)
        try:
            os.mkdir(self.save_path)
            print(' '.join(['Simulation.generate_path :: Created ', self.save_path]))
        except:
            raise('Simulation.generate_path :: Path creation failed.')

    # TODO: Finish this method
    def validate_settings(self):
        self.is_valid = True


def parse_input_yaml(input_path:str) -> dict:
    try:
        with open(input_path, 'r') as stream:
            yaml_content_dictionary = yaml.safe_load(stream)
            stream.close()
        return yaml_content_dictionary
    except:
        return {}


def generate_simulations(elements:dict, max_number:int = 10):
    # Check what to randomize and Combine
    rand_var_list = elements['Randomize']
    if 'None' in rand_var_list:
        rand_var_list = []
    comb_var_list = elements['Combine']
    if 'None' in comb_var_list:
        comb_var_list = []
    # Create Base Simulation
    base_sim = Simulation(elements['Settings'])
    if comb_var_list == [] and rand_var_list == []:
        print('generate_simulations :: Only one simulation requested!')
        return [base_sim]
    # Create the random sims
    if rand_var_list != []:
        random_indices = {}
        for item in rand_var_list:
            # Create more than needed to keep unique only
            random_indices[item] = np.random.randint(0, len(elements['Elements'][item]), size=(1, 2 * max_number))
        # Create all random runs
        simulation_list_rand = []
        for i in range(2 * max_number):
            new_random_sim = copy.deepcopy(base_sim)
            for item in rand_var_list:
                setattr(new_random_sim, new_random_sim.attribute_dictionary[item], elements['Elements'][item][random_indices[item][0,i]])
            setattr(new_random_sim, 'full_string', new_random_sim.assemble_fullstring(new_random_sim.mood, new_random_sim.subject, new_random_sim.style))
            simulation_list_rand.append(new_random_sim)
        # TODO: Check for uniqueness and reduce the number of sims to max_number
        simulation_list_rand = list(set(simulation_list_rand[0:max_number]))
    else:
        simulation_list_rand = [base_sim]
    # Create the combine sims
    if comb_var_list != []:
        simulation_list_comb = []
        # Define the combinations
        max_dimensions = [np.arange(len(elements['Elements'][varname])) for varname in comb_var_list]
        # Apply combinations to all sims
        for sim in simulation_list_rand:
            combinations = itertools.product(*max_dimensions)
            for combo in combinations:
                new_combo_sim = copy.deepcopy(sim)
                for index, varname in enumerate(comb_var_list):
                    setattr(new_combo_sim, new_combo_sim.attribute_dictionary[varname], elements['Elements'][varname][combo[index]])
                simulation_list_comb.append(new_combo_sim)
        return simulation_list_comb
    else:
        simulation_list_comb = simulation_list_rand
        return simulation_list_comb

if __name__ == "__main__":
    print(''.join(['custom_parser :: Loading file : ', input_path]))
    settings_dictionary = parse_input_yaml(input_path)
    list_of_sims = generate_simulations(settings_dictionary, max_number = 5)
    for sim in list_of_sims:
        print('------------------')
        print('String : ' + sim.full_string)
        if sim.initial_image != []:
            for img in sim.initial_image:
                print('Initial Image / weight: ' + img + ' / ' + str(sim.initial_image_weight))
        else:
            print('No initial image is present.')
        print('Seed : ' + str(sim.seed))
        


