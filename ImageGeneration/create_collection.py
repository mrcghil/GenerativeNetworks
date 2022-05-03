import os
# Collection origin
COLLECTION_PATH = 'C:\\WORKSPACES\\ZINKY\\GenerativeNetworks\\Results\\NamedCollection01'
list_of_images = [item for item in os.listdir(COLLECTION_PATH) if os.path.isfile(os.path.join(COLLECTION_PATH, item))]



if __name__ == '__main__':

for nft in list_of_images:
    pass