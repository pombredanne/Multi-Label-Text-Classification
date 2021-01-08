import numpy as np
# from sentence_transformers import SentenceTransformer, util

from config import *

label_definition_list = [
    ['commercial_use', 'use the software for commercial purposes'],
    ['distribution', 'distribute original or modified or derivative works'],
    ['modification', 'modify the software and or create derivative works'],
    ['private_use', 'use or modify the software freely and privately'],
    ['patent_use', 'practice patent claims of contributors to the software'],
    ['trademark_use', 'use the names, trademarks or logos of contributors'],
    ['disclose_source', 'disclose or provide source code when distributing the software'],
    # ['network use disclose', 'disclose or provide source code when distributing the software through network or web medium'],
    ['license_and_copyright_notice', 'include the license file and or the copyright notice when distributing the original or copy of the software'],
    ['same_license', 'the modified or altered or derived versions of the software must be distributed under the license and or the copyright notice of the original software'],
    ['state_changes', 'carry notice stating significant changes that made to the software'],
    ['liability', 'disclaimer of liability, whether the software owner or contributor should be charged for damages'],
    ['warranty', 'disclaimer of warranty, whether the software provides any kind of warranties']
]

label_definition_dict = {
    'commercial use': 'use the software for commercial purposes',
    'distribution': 'distribute original or modified or derivative works',
    'modification': 'modify the software and or create derivative works',
    'private use': 'use or modify the software freely and privately',
    'patent use': 'practice patent claims of contributors to the software',
    'trademark use': 'use the names, trademarks or logos of contributors',
    'disclose source': 'disclose or provide source code when distributing the software',
    # 'network use disclose': 'disclose or provide source code when distributing the software through network or web medium',
    'licence and copyright notice': 'include the license file and or the copyright notice '
                                    'when distributing the original or copy of the software',
    'same license': 'the modified or altered or derived versions of the software must be distributed '
                    'under the license and or the copyright notice of the original software',
    'state changes': 'carry notice stating significant changes that made to the software',
    'liability': 'disclaimer of liability, whether the software owner or contributor should be charged for damages',
    'warranty': 'disclaimer of warranty, whether the software provides any kind of warranties'
}


if __name__ == '__main__':

    # Initialize Sentence Transformer
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')

    result_array = np.ones(shape=(num_of_class, label_embedding_size), dtype=float)
    result_array[0] = model.encode(label_definition_dict['commercial use'], convert_to_tensor=False, device='cpu')
    # print(result_array[0])
    result_array[1] = model.encode(label_definition_dict['distribution'], convert_to_tensor=False, device='cpu')
    # print(result_array[1])
    result_array[2] = model.encode(label_definition_dict['modification'], convert_to_tensor=False, device='cpu')
    # print(result_array[2])
    result_array[3] = model.encode(label_definition_dict['private use'], convert_to_tensor=False, device='cpu')
    # print(result_array[3])
    result_array[4] = model.encode(label_definition_dict['patent use'], convert_to_tensor=False, device='cpu')
    # print(result_array[4])
    result_array[5] = model.encode(label_definition_dict['trademark use'], convert_to_tensor=False, device='cpu')
    # print(result_array[5])
    result_array[6] = model.encode(label_definition_dict['disclose source'], convert_to_tensor=False, device='cpu')
    # print(result_array[6])
    result_array[7] = model.encode(label_definition_dict['network use disclose'], convert_to_tensor=False, device='cpu')
    # print(result_array[7])
    result_array[8] = model.encode(label_definition_dict['licence and copyright notice'], convert_to_tensor=False, device='cpu')
    # print(result_array[8])
    result_array[9] = model.encode(label_definition_dict['same license'], convert_to_tensor=False, device='cpu')
    # print(result_array[9])
    result_array[10] = model.encode(label_definition_dict['state changes'], convert_to_tensor=False, device='cpu')
    # print(result_array[10])
    result_array[11] = model.encode(label_definition_dict['liability'], convert_to_tensor=False, device='cpu')
    # print(result_array[11])
    result_array[12] = model.encode(label_definition_dict['warranty'], convert_to_tensor=False, device='cpu')
    print(result_array[12])
    # np.save('./cache/label_embeddings.npy', result_array)
    # print('label_embeddings.npy save done')
    arr = np.load('./cache/label_embeddings.npy')
    print(arr[12])
    print('label_embeddings.npy load done')
