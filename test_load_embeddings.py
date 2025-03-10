import fetch_embeddings
import torch

_MPS_DEVICE = torch.device("mps")
_EMBEDDINGS_FILES = ['./data/embeddings_1of2.txt',
					 './data/embeddings_2of2.txt']
_PASSAGE_PATH = './data/combined_data.jsonl'
_BUFFER_SIZE = 1_000

def load_embeddings(files: list[str]) -> torch.Tensor:
	embeddings = None
	buffer = None
	for path in files:
		with open(path, 'r') as f:
			lines = f.readlines()
			for i, line in enumerate(lines):
				l = line.strip()[1: -1]  # exclude [ ]
				l = l.split(',')
				l = [float(num) for num in l]
				t = torch.Tensor([l])
				# t = t.to(_MPS_DEVICE)
				if buffer == None:
					buffer = t
				else:
					buffer = torch.cat([buffer, t], axis=0)
				if i % _BUFFER_SIZE == 0:
					buffer = buffer.to(_MPS_DEVICE)
					if embeddings == None:
						embeddings = buffer
					else:
						embeddings = torch.cat([embeddings, buffer], axis=0)
					del buffer
					buffer = None
					print(embeddings.shape)
	return embeddings


_EMBEDDINGS = load_embeddings(_EMBEDDINGS_FILES[:1])
print(_EMBEDDINGS.shape)
print(_EMBEDDINGS.device)

# client = fetch_embeddings.get_openai_client()

# query_embedding = fetch_embeddings.get_embedding(client, 0, 'who won wimbledon in 2019?')
 
# print(query_embedding)




