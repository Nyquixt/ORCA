import torch
import torch.nn as nn
import torch.nn.functional as F
    
class ConfidenceModel(nn.Module):
    def __init__(self, method, clip_model, description_encodings, n_concepts, n_classes, temperature=1.0) -> None:
        super().__init__()
        assert method in ['base', 'rank']
        self.method = method
        self.clip_model = clip_model
        self.description_encodings = description_encodings
        self.n_concepts = n_concepts
        self.n_classes = n_classes
        self.temperature = temperature

    def predict(self, images):
        image_encodings = self.clip_model.encode_image(images)
        image_encodings = F.normalize(image_encodings, dim=-1)
        image_description_similarity = [None] * self.n_classes
    
        for i, (k, v) in enumerate(self.description_encodings.items()): # You can also vectorize this; it wasn't much faster for me
            # print(image_encodings.dtype, v.dtype)
            dot_product_matrix = image_encodings @ v.T
            image_description_similarity[i] = dot_product_matrix

        image_description_similarity = torch.stack(image_description_similarity, dim=1).flatten(start_dim=1) # batch x n_concepts*n_classes
        if self.method == 'base':
            predictions, confidences = self.get_base_outputs(image_description_similarity)
        else:
            predictions, confidences = self.get_rank_outputs(image_description_similarity)
        return predictions, confidences, image_description_similarity

    # ORCA-B
    def get_base_outputs(self, all_similarities):
        # all_similarities: # batch x n_concepts*n_classes
        # n_concepts: num concepts per class
        predictions = torch.zeros((all_similarities.shape[0],), dtype=torch.float32)
        confidences = torch.zeros((all_similarities.shape[0],), dtype=torch.float32)

        for i in range(confidences.shape[0]):
            concepts_similarity = all_similarities[i]
            descend_topk_indices = torch.flip(concepts_similarity.argsort(), dims=[0])[:self.n_concepts]
            concept_of_class = descend_topk_indices // self.n_concepts
            # count the number of concepts belonging to each class in the top-k
            unique, counts = torch.unique(concept_of_class, return_counts=True)
            count_dict = dict(zip(unique, counts))
            prediction = max(count_dict, key= lambda x: count_dict[x])
            
            unmatched = (concept_of_class != prediction).type(torch.int8) 
            confidences[i] = 1. - (torch.sum(unmatched) / float(self.n_concepts))
            predictions[i] = prediction

        return predictions, confidences
    
    def linear_weights(self):
        ranks = torch.arange(self.n_concepts, 0, -1) # Assign ranks in descending order
        return ranks / torch.sum(ranks)

    def exponential_weights(self):
        ranks = torch.arange(self.n_concepts, 0, -1) # Assign ranks in descending order
        values = torch.exp(ranks / self.temperature) # Calculate values using softmax-like function
        values /= torch.sum(values) # Normalize the values to sum up to 1
        return values
    
    def log_scale_weights(self):
        ranks = torch.arange(self.n_concepts, 0, -1) # Generate a rank vector from K to 1
        log_scaled_weights = torch.log1p(ranks)  # Apply log scaling. Using log1p for numerical stability
        normalized_weights = log_scaled_weights / torch.sum(log_scaled_weights) # Normalize weights to sum to 1
        return normalized_weights
    
    def polynomial_weights(self, exponent=2):
        ranks = torch.arange(self.n_concepts, 0, -1) # Assign ranks in descending order
        poly_weights = ranks ** exponent / torch.sum(ranks ** exponent)
        return poly_weights

    # ORCA-R
    def get_rank_outputs(self, all_similarities):
        # all_similarities: # batch x n_concepts*n_classes
        # n_concepts: num concepts per class
        predictions = torch.zeros((all_similarities.size(0),), dtype=torch.float32)
        confidences = torch.zeros((all_similarities.size(0),), dtype=torch.float32)
        # Logarithmic
        ranking_vector = self.log_scale_weights().to(all_similarities.device)
        # Polynomial
        # ranking_vector = self.polynomial_weights().to(all_similarities.device)
        # Exponential
        # ranking_vector = self.exponential_weights().to(all_similarities.device)
        # Linear
        # ranking_vector = self.linear_weights().to(all_similarities.device)
        for i in range(confidences.shape[0]):
            concepts_similarity = all_similarities[i]
            descend_topk_indices = torch.flip(concepts_similarity.argsort(), dims=[0])[:self.n_concepts]
            concept_of_class = descend_topk_indices // self.n_concepts
            # count the number of concepts belonging to each class in the top-k
            unique = torch.unique(concept_of_class)

            confs = []
            for item in unique:
                binary_vector = (concept_of_class == item).type(torch.int16)
                confs.append(torch.sum(binary_vector * ranking_vector))
            
            conf_dict = dict(zip(unique, confs))
            prediction = max(conf_dict, key= lambda x: conf_dict[x])
            
            confidences[i] = conf_dict[prediction]
            predictions[i] = prediction

        return predictions, confidences