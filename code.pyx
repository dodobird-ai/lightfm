""" 
Extracts from lightfm/lightfm/_lightfm_fast.pyx.template
for chatGPT to have a global view & understanding of action & interactions
"""

#!python
#cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False

import numpy as np

cimport cython.operator.dereference as deref
from libc.stdlib cimport free, malloc

{openmp_import}


ctypedef float flt

# Allow sequential code blocks in a parallel setting.
# Used for applying full regularization in parallel blocks.
{lock_init}


cdef flt MAX_REG_SCALE = 1000000.0


cdef extern from "math.h" nogil:
    double sqrt(double)
    double exp(double)
    double log(double)
    double floor(double)


cdef extern from "stdlib.h" nogil:
    ctypedef void const_void "const void"
    void qsort(void *base, int nmemb, int size,
               int(*compar)(const_void *, const_void *)) nogil
    void* bsearch(const void *key, void *base, int nmemb, int size,
                  int(*compar)(const_void *, const_void *)) nogil



cdef inline flt compute_prediction_from_repr(flt *user_repr,
                                             flt *item_repr,
                                             int no_components) nogil:

    cdef int i
    cdef flt result

    # Biases
    result = user_repr[no_components] + item_repr[no_components]

    # Latent factor dot product
    for i in range(no_components):
        result += user_repr[i] * item_repr[i]

    return result

cdef void locked_regularize(FastLightFM lightfm,
                            double item_alpha,
                            double user_alpha) nogil:
    """
    Apply accumulated L2 regularization to all features. Acquire a lock
    to prevent multiple threads from performing this operation.
    """

    {lock_acquire}
    if lightfm.item_scale > MAX_REG_SCALE or lightfm.user_scale > MAX_REG_SCALE:
        regularize(lightfm,
                   item_alpha,
                   user_alpha)
    {lock_release}


cdef inline void compute_representation(CSRMatrix features,
                                        flt[:, ::1] feature_embeddings,
                                        flt[::1] feature_biases,
                                        FastLightFM lightfm,
                                        int row_id,
                                        double scale,
                                        flt *representation) nogil:
    """
    Compute latent representation for row_id.
    The last element of the representation is the bias.
    """

    cdef int i, j, start_index, stop_index, feature
    cdef flt feature_weight

    start_index = features.get_row_start(row_id)
    stop_index = features.get_row_end(row_id)

    for i in range(lightfm.no_components + 1):
        representation[i] = 0.0

    for i in range(start_index, stop_index):

        feature = features.indices[i]
        feature_weight = features.data[i] * scale

        for j in range(lightfm.no_components):

            representation[j] += feature_weight * feature_embeddings[feature, j]

        representation[lightfm.no_components] += feature_weight * feature_biases[feature]

cdef void warp_update(double loss,
                      CSRMatrix item_features,
                      CSRMatrix user_features,
                      int user_id,
                      int positive_item_id,
                      int negative_item_id,
                      flt *user_repr,
                      flt *pos_it_repr,
                      flt *neg_it_repr,
                      FastLightFM lightfm,
                      double item_alpha,
                      double user_alpha) nogil:
    """
    Apply the gradient step.
    """

    cdef int i, j, positive_item_start_index, positive_item_stop_index
    cdef int  user_start_index, user_stop_index, negative_item_start_index, negative_item_stop_index
    cdef double avg_learning_rate
    cdef flt positive_item_component, negative_item_component, user_component

    avg_learning_rate = 0.0

    # Get the iteration ranges for features
    # for this training example.
    positive_item_start_index = item_features.get_row_start(positive_item_id)
    positive_item_stop_index = item_features.get_row_end(positive_item_id)

    negative_item_start_index = item_features.get_row_start(negative_item_id)
    negative_item_stop_index = item_features.get_row_end(negative_item_id)

    user_start_index = user_features.get_row_start(user_id)
    user_stop_index = user_features.get_row_end(user_id)

    avg_learning_rate += update_biases(item_features, positive_item_start_index,
                                       positive_item_stop_index,
                                       lightfm.item_biases, lightfm.item_bias_gradients,
                                       lightfm.item_bias_momentum,
                                       -loss,
                                       lightfm.adadelta,
                                       lightfm.learning_rate,
                                       item_alpha,
                                       lightfm.rho,
                                       lightfm.eps)

    avg_learning_rate += update_biases(item_features, negative_item_start_index,
                                       negative_item_stop_index,
                                       lightfm.item_biases, lightfm.item_bias_gradients,
                                       lightfm.item_bias_momentum,
                                       loss,
                                       lightfm.adadelta,
                                       lightfm.learning_rate,
                                       item_alpha,
                                       lightfm.rho,
                                       lightfm.eps)

    avg_learning_rate += update_biases(user_features, user_start_index, user_stop_index,
                                       lightfm.user_biases, lightfm.user_bias_gradients,
                                       lightfm.user_bias_momentum,
                                       loss,
                                       lightfm.adadelta,
                                       lightfm.learning_rate,
                                       user_alpha,
                                       lightfm.rho,
                                       lightfm.eps)

    # Update latent representations.
    for i in range(lightfm.no_components):

        user_component = user_repr[i]
        positive_item_component = pos_it_repr[i]
        negative_item_component = neg_it_repr[i]

        avg_learning_rate += update_features(item_features, lightfm.item_features,
                                             lightfm.item_feature_gradients,
                                             lightfm.item_feature_momentum,
                                             i, positive_item_start_index, positive_item_stop_index,
                                             -loss * user_component,
                                             lightfm.adadelta,
                                             lightfm.learning_rate,
                                             item_alpha,
                                             lightfm.rho,
                                             lightfm.eps)

        avg_learning_rate += update_features(item_features, lightfm.item_features,
                                             lightfm.item_feature_gradients,
                                             lightfm.item_feature_momentum,
                                             i, negative_item_start_index, negative_item_stop_index,
                                             loss * user_component,
                                             lightfm.adadelta,
                                             lightfm.learning_rate,
                                             item_alpha,
                                             lightfm.rho,
                                             lightfm.eps)

        avg_learning_rate += update_features(user_features, lightfm.user_features,
                                             lightfm.user_feature_gradients,
                                             lightfm.user_feature_momentum,
                                             i, user_start_index, user_stop_index,
                                             loss * (negative_item_component -
                                                     positive_item_component),
                                             lightfm.adadelta,
                                             lightfm.learning_rate,
                                             user_alpha,
                                             lightfm.rho,
                                             lightfm.eps)

    avg_learning_rate /= ((lightfm.no_components + 1) * (user_stop_index - user_start_index)
                          + (lightfm.no_components + 1) *
                          (positive_item_stop_index - positive_item_start_index)
                          + (lightfm.no_components + 1)
                          * (negative_item_stop_index - negative_item_start_index))

    # Update the scaling factors for lazy regularization, using the average learning rate
    # of features updated for this example.
    lightfm.item_scale *= (1.0 + item_alpha * avg_learning_rate)
    lightfm.user_scale *= (1.0 + user_alpha * avg_learning_rate)

    lightfm.avg_loss_ctr += 1
    lightfm.avg_loss = (lightfm.avg_loss * (lightfm.avg_loss_ctr - 1) + loss) / lightfm.avg_loss_ctr



def fit_warp(CSRMatrix item_features,
             CSRMatrix user_features,
             CSRMatrix interactions,
             int[::1] user_ids,
             int[::1] item_ids,
             flt[::1] Y,
             flt[::1] sample_weight,
             int[::1] shuffle_indices,
             FastLightFM lightfm,
             double learning_rate,
             double item_alpha,
             double user_alpha,
             int num_threads,
             random_state):
    """
    Fit the model using the WARP loss.
    """

    cdef int i, no_examples, user_id, positive_item_id, gamma
    cdef int negative_item_id, sampled, row
    cdef double positive_prediction, negative_prediction
    cdef double loss, MAX_LOSS
    cdef flt weight
    cdef flt *user_repr
    cdef flt *pos_it_repr
    cdef flt *neg_it_repr
    cdef unsigned int[::1] random_states

    random_states = random_state.randint(0,
                                         np.iinfo(np.int32).max,
                                         size=num_threads).astype(np.uint32)

    no_examples = Y.shape[0]
    MAX_LOSS = 10.0

    {nogil_block}

        user_repr = <flt *>malloc(sizeof(flt) * (lightfm.no_components + 1))
        pos_it_repr = <flt *>malloc(sizeof(flt) * (lightfm.no_components + 1))
        neg_it_repr = <flt *>malloc(sizeof(flt) * (lightfm.no_components + 1))

        for i in {range_block}(no_examples):
            row = shuffle_indices[i]

            user_id = user_ids[row]
            positive_item_id = item_ids[row]

            if not Y[row] > 0:
                continue

            weight = sample_weight[row]

            compute_representation(user_features,
                                   lightfm.user_features,
                                   lightfm.user_biases,
                                   lightfm,
                                   user_id,
                                   lightfm.user_scale,
                                   user_repr)

            compute_representation(item_features,
                                   lightfm.item_features,
                                   lightfm.item_biases,
                                   lightfm,
                                   positive_item_id,
                                   lightfm.item_scale,
                                   pos_it_repr)

            positive_prediction = compute_prediction_from_repr(user_repr,
                                                               pos_it_repr,
                                                               lightfm.no_components)

            sampled = 0

            while sampled < lightfm.max_sampled:

                sampled = sampled + 1
                negative_item_id = (rand_r(&random_states[{thread_num}])
                                    % item_features.rows)

                compute_representation(item_features,
                                       lightfm.item_features,
                                       lightfm.item_biases,
                                       lightfm,
                                       negative_item_id,
                                       lightfm.item_scale,
                                       neg_it_repr)

                negative_prediction = compute_prediction_from_repr(user_repr,
                                                                   neg_it_repr,
                                                                   lightfm.no_components)

                if negative_prediction > positive_prediction - 1:

                    # Sample again if the sample negative is actually a positive
                    if in_positives(negative_item_id, user_id, interactions):
                        continue

                    loss = weight * log(max(1.0, floor((item_features.rows - 1) / sampled)))

                    # Clip gradients for numerical stability.
                    if loss > MAX_LOSS:
                        loss = MAX_LOSS

                    warp_update(loss,
                                item_features,
                                user_features,
                                user_id,
                                positive_item_id,
                                negative_item_id,
                                user_repr,
                                pos_it_repr,
                                neg_it_repr,
                                lightfm,
                                item_alpha,
                                user_alpha)
                    break

            if lightfm.item_scale > MAX_REG_SCALE or lightfm.user_scale > MAX_REG_SCALE:
                locked_regularize(lightfm,
                                  item_alpha,
                                  user_alpha)

        free(user_repr)
        free(pos_it_repr)
        free(neg_it_repr)

    regularize(lightfm,
               item_alpha,
               user_alpha)


cdef double update_biases(CSRMatrix feature_indices,
                          int start,
                          int stop,
                          flt[::1] biases,
                          flt[::1] gradients,
                          flt[::1] momentum,
                          double gradient,
                          int adadelta,
                          double learning_rate,
                          double alpha,
                          flt rho,
                          flt eps) nogil:
    """
    Perform a SGD update of the bias terms.
    """

    cdef int i, feature
    cdef double feature_weight, local_learning_rate, sum_learning_rate, update

    sum_learning_rate = 0.0

    if adadelta:
        for i in range(start, stop):

            feature = feature_indices.indices[i]
            feature_weight = feature_indices.data[i]

            gradients[feature] = rho * gradients[feature] + (1 - rho) * (feature_weight * gradient) ** 2
            local_learning_rate = sqrt(momentum[feature] + eps) / sqrt(gradients[feature] + eps)
            update = local_learning_rate * gradient * feature_weight
            momentum[feature] = rho * momentum[feature] + (1 - rho) * update ** 2
            biases[feature] -= update

            # Lazy regularization: scale up by the regularization
            # parameter.
            biases[feature] *= (1.0 + alpha * local_learning_rate)

            sum_learning_rate += local_learning_rate
    else:
        for i in range(start, stop):

            feature = feature_indices.indices[i]
            feature_weight = feature_indices.data[i]

            local_learning_rate = learning_rate / sqrt(gradients[feature])
            biases[feature] -= local_learning_rate * feature_weight * gradient
            gradients[feature] += (gradient * feature_weight) ** 2

            # Lazy regularization: scale up by the regularization
            # parameter.
            biases[feature] *= (1.0 + alpha * local_learning_rate)

            sum_learning_rate += local_learning_rate

    return sum_learning_rate


cdef inline double update_features(CSRMatrix feature_indices,
                                   flt[:, ::1] features,
                                   flt[:, ::1] gradients,
                                   flt[:, ::1] momentum,
                                   int component,
                                   int start,
                                   int stop,
                                   double gradient,
                                   int adadelta,
                                   double learning_rate,
                                   double alpha,
                                   flt rho,
                                   flt eps) nogil:
    """
    Update feature vectors.
    """

    cdef int i, feature,
    cdef double feature_weight, local_learning_rate, sum_learning_rate, update

    sum_learning_rate = 0.0

    if adadelta:
        for i in range(start, stop):

            feature = feature_indices.indices[i]
            feature_weight = feature_indices.data[i]

            gradients[feature, component] = (rho * gradients[feature, component]
                                             + (1 - rho) * (feature_weight * gradient) ** 2)
            local_learning_rate = (sqrt(momentum[feature, component] + eps)
                                   / sqrt(gradients[feature, component] + eps))
            update = local_learning_rate * gradient * feature_weight
            momentum[feature, component] = rho * momentum[feature, component] + (1 - rho) * update ** 2
            features[feature, component] -= update

            # Lazy regularization: scale up by the regularization
            # parameter.
            features[feature, component] *= (1.0 + alpha * local_learning_rate)

            sum_learning_rate += local_learning_rate
    else:
        for i in range(start, stop):

            feature = feature_indices.indices[i]
            feature_weight = feature_indices.data[i]

            local_learning_rate = learning_rate / sqrt(gradients[feature, component])
            features[feature, component] -= local_learning_rate * feature_weight * gradient
            gradients[feature, component] += (gradient * feature_weight) ** 2

            # Lazy regularization: scale up by the regularization
            # parameter.
            features[feature, component] *= (1.0 + alpha * local_learning_rate)

            sum_learning_rate += local_learning_rate

    return sum_learning_rate
