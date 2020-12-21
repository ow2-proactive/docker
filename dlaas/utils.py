import dlaas_pb2
import sys
import time
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array


#Read the existing binary configuration file
#or create a new one
def read_config_file():
	print("Reading the configuration file of Tensorflow service ")
	ModelConfigList_instance = dlaas_pb2.ModelConfigList()
	try:
  		f = open("config_file.bin", "rb")
  		ModelConfigList_instance.ParseFromString(f.read())
  		f.close()
	except IOError:
  		print ("[INFO]config_file.bin: Could not open file.  Creating a new one.")
  		f = open("config_file.bin", "wb")
  		f.close()
	return ModelConfigList_instance

#update the binary configuration file and the txt configuration file
def update_model_service_config_file(config_list):
	print("[INFO] Updating the configuration file of Tensorflow service ")
	f = open("config_file.bin", "wb")
	f.write(config_list.SerializeToString())
	f.close()
	file_config = dlaas_pb2.FileConfig()
	file_config.model_config_list.append(config_list)
	print(file_config)
	f = open("models.config", "w+")
	f.write(str(file_config))
	f.close()

#add a specific model version to be deployed in the congig file
def add_version_model_service_config(model_name, model_path = "/tmp", version = None) -> str:
	print("Adding the Model_service ", model_name)
	ModelConfigList_instance = dlaas_pb2.ModelConfigList()
	#ModelConfigList_instance = read_config_file()
	model_config = dlaas_pb2.ModelConfig()
	model_config.name = model_name
	model_config.base_path = model_path
	model_config.model_platform = "tensorflow"
	specific_instance = dlaas_pb2.Specific()
	specific_instance.versions.extend([version])
	model_version_policy_instance = model_config.model_version_policy.add()
	model_version_policy_instance.specific.append(specific_instance)
	model_version_policy_instance_test = model_config.model_version_policy
	ModelConfigList_instance.config.append(model_config)
	update_model_service_config_file(ModelConfigList_instance)
	print("[INFO] The version ", version, "of the model ", model_name, " was successefully deployed")
	add_status = "The version " + str(version) + " of the model " + str(model_name) + " was successefully deployed"
	return add_status

#append a specific model version to be deployed in the congig file
def append_version_model_service_config(model_name, model_path = "/tmp", version = None) -> str:
	print("Appending the Model_service ", model_name)
	current_model_config = read_config_file()
	model_name_exists = False
	version_exists = False
	for model_config in current_model_config.config:
		if (model_config.name == model_name):
			print("[WARN] This service name already exists. Its configuration will be updated")
			model_name_exists = True
			for version_policy in model_config.model_version_policy:
				for specific_sample in version_policy.specific:
					for version_sample in specific_sample.versions:
						if version_sample == version:
							version_exists = True
							print("[WARN] The version ", version, " of the model ", model_name, " is already deployed")
							append_status = "The version " + str(version) + " of the model " + str(model_name) + " is already deployed"
					if not version_exists:
						specific_sample.versions.extend([version])
						update_model_service_config_file(current_model_config)
						print("[INFO] The version ", version, " of the model ", model_name, " was successefully deployed")
						append_status = "The version " + str(version) + " of the model " + str(model_name) + " was successefully deployed"
	if not model_name_exists:
		model_config = dlaas_pb2.ModelConfig()
		model_config.name = model_name
		model_config.base_path = model_path
		model_config.model_platform = "tensorflow"
		specific_instance = dlaas_pb2.Specific()
		specific_instance.versions.extend([version])
		model_version_policy_instance = model_config.model_version_policy.add()
		model_version_policy_instance.specific.append(specific_instance)
		model_version_policy_instance_test = model_config.model_version_policy
		current_model_config.config.append(model_config)
		update_model_service_config_file(current_model_config)
		print("[INFO] The version ", version, " of the model ", model_name, " was successefully deployed")
		append_status = "The version " + str(version) + " of the model " + str(model_name) + " was successefully deployed"
	return append_status

#delete a specific model version to be deployed in the congig file
def delete_version_model_service_config(model_name, version=None) -> str:
	print("Removing the Model_service ", model_name)
	ModelConfigList_instance = dlaas_pb2.ModelConfigList()
	ModelConfigList_instance_updated = dlaas_pb2.ModelConfigList()
	ModelConfigList_instance = read_config_file()
	model_name_exists = False
	version_exists = False
	for model_config in ModelConfigList_instance.config:
		if (model_config.name == model_name):
			print("[INFO] Model_service : ", model_name, " found")
			model_name_exists = True
			if version is not None:
				for version_policy in model_config.model_version_policy:
					for specific_sample in version_policy.specific:
						count_versions = 0
						for version_sample in specific_sample.versions:
							count_versions = count_versions + 1
							if version_sample == version:
								version_exists = True
						if version_exists and count_versions>1:
							specific_sample.versions.remove(version)
							print("[INFO] Undeploying the version ", version, "of the model ", model_name)
							delete_status = "The version " + str(version) + " of the model " + str(model_name) + " was successefully undeployed"
						elif version_exists and count_versions == 1:
							delete_status = "The Model " + str(model_name) + " was successefully undeployed"
						else:
							print("[WARN] The version ", version, "of the model ", model_name, " is not deployed")
							delete_status = "The version " + str(version) + " of the model " + str(model_name) + " is not deployed. Please specify an already deployed version"
			else:
				print("[INFO] Undeploying the model ", model_name)
				delete_status = "The Model " + str(model_name) + " was successefully undeployed"
		else:
			ModelConfigList_instance_updated.config.append(model_config)
	if not model_name_exists:
		delete_status = "The Model " + str(model_name) + " is not deployed. Please specify an already deployed model"
		print("[WARN] Model_service ", model_name ," doesn't exist")
	elif version_exists and count_versions>1:
		update_model_service_config_file(ModelConfigList_instance)
	else:
		update_model_service_config_file(ModelConfigList_instance_updated)
	return delete_status

#check if a specific model version exists in the config file
def check_model_name_version(model_name,version) -> str:
	print("Checking model_name version ", model_name)
	ModelConfigList_instance = dlaas_pb2.ModelConfigList()
	ModelConfigList_instance_updated = dlaas_pb2.ModelConfigList()
	ModelConfigList_instance = read_config_file()
	model_name_deployed = False
	version_deployed = False
	for model_config in ModelConfigList_instance.config:
		if (model_config.name == model_name):
			print("[INFO] Model_service : ", model_name, " found")
			model_name_deployed = True
			if version is not None:
				for version_policy in model_config.model_version_policy:
					for specific_sample in version_policy.specific:
						for version_sample in specific_sample.versions:
							if version_sample == version:
									version_deployed = True
	if not model_name_deployed:
		status = "The model " + str(model_name) + " is not deployed. Please specify an already deployed model."
	elif model_name_deployed and not version_deployed:
		status = "The version " + str(version) + " of the model " + str(model_name) + " is not deployed. Please specify an already deployed version."
	elif version is None and model_name_deployed:
		status = "model_name_deployed"
	else:
		status = "version deployed"
	return status

#check if a specific model exists in the config file
def check_deployed_model_name_version(model_name,version) -> bool:
	print("Checking model_name version ", model_name)
	ModelConfigList_instance = dlaas_pb2.ModelConfigList()
	ModelConfigList_instance_updated = dlaas_pb2.ModelConfigList()
	ModelConfigList_instance = read_config_file()
	deployed = False
	for model_config in ModelConfigList_instance.config:
		if (model_config.name == model_name):
			model_name_deployed = True
			if version is not None:
				for version_policy in model_config.model_version_policy:
					for specific_sample in version_policy.specific:
						for version_sample in specific_sample.versions:
							if version_sample == version:
									deployed = True
			else:
				deployed = True
	return deployed

#returns the newest deployed version
def get_newest_deployed_version(model_name) -> int:
	print("Removing the Model_service ", model_name)
	ModelConfigList_instance = dlaas_pb2.ModelConfigList()
	ModelConfigList_instance_updated = dlaas_pb2.ModelConfigList()
	ModelConfigList_instance = read_config_file()
	model_name_deployed = False
	max = 0
	for model_config in ModelConfigList_instance.config:
		if (model_config.name == model_name):
			print("[INFO] Model_service : ", model_name, " found")
			model_name_deployed = True
			for version_policy in model_config.model_version_policy:
				for specific_sample in version_policy.specific:
					for version_sample in specific_sample.versions:
						if version_sample>max :
							max = version_sample
	return max	

# load and prepare the image
def load_image(filename):
	# load the image
	img = load_img(filename, color_mode="grayscale", target_size=(28, 28))
	# convert to array
	img = img_to_array(img)
	print("img")
	# reshape into a single sample with 1 channel
	img = img.reshape(1, 28, 28, 1)
	# prepare pixel data
	img = img.astype('float32')
	img = img / 255.0
	#print ("img :", img)
	return img					
