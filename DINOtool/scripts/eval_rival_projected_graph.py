"""Frozen transfer-selected candidate; original full-suite model remains separate."""
import torch

from dinotool.rival_competition_admission import IMPLEMENTATION, settings
from eval_rival_competition_transfer import CANDIDATE, bind, predict_image
import eval_rival_fine_full as reference


METHODS = ('Geometry', 'NoAdmission_Exact', 'RivalFineHard_Exact', CANDIDATE)


def predictor(*inputs):
    return predict_image(*inputs, methods=METHODS)


def save(path, result):
    result['signature']['gear']['competition_action'] = settings()
    reference.save(path, dict(result,
        execution_backend='cached competition reader + original batch-one512 fine CUDA graph',
        candidate_status='frozen development/transfer-selected; not the established full-suite model'))


if __name__ == '__main__':
    with torch.inference_mode():
        bind(reference.main, predict_image=predictor, METHODS=METHODS,
             IMPLEMENTATION=IMPLEMENTATION, save=save)(reference.parse_args())
