import torch

from scripts.evaluation.build_grpo_model_soup import average_decoder_state_dicts


DECODER = "agent._transfuser_model._trajectory_head.diff_decoder."


def test_average_decoder_uses_base_for_frozen_weights_and_drops_snapshots():
    base = {
        "agent._transfuser_model._backbone.weight": torch.tensor([10.0]),
        f"{DECODER}weight": torch.tensor([0.0, 0.0]),
        f"{DECODER}counter": torch.tensor(3, dtype=torch.long),
    }
    source_a = {
        **base,
        "agent._transfuser_model._backbone.weight": torch.tensor([99.0]),
        f"{DECODER}weight": torch.tensor([1.0, 3.0]),
        "agent._transfuser_model._trajectory_head.old_policy.weight": torch.tensor([5.0]),
    }
    source_b = {
        **base,
        "agent._transfuser_model._backbone.weight": torch.tensor([-99.0]),
        f"{DECODER}weight": torch.tensor([3.0, 5.0]),
        "agent._transfuser_model._trajectory_head.ref_policy.weight": torch.tensor([7.0]),
    }

    output, counts = average_decoder_state_dicts(base, [source_a, source_b])

    assert torch.equal(
        output["_transfuser_model._trajectory_head.diff_decoder.weight"],
        torch.tensor([2.0, 4.0]),
    )
    assert torch.equal(
        output["_transfuser_model._backbone.weight"], torch.tensor([10.0])
    )
    assert not any("old_policy" in key or "ref_policy" in key for key in output)
    assert counts["averaged_tensors"] == 1
    assert counts["averaged_numel"] == 2


def test_average_decoder_requires_matching_keys():
    base = {f"{DECODER}weight": torch.tensor([0.0])}
    mismatched = {f"{DECODER}other": torch.tensor([1.0])}

    try:
        average_decoder_state_dicts(base, [mismatched, mismatched])
    except ValueError as error:
        assert "decoder keys mismatch" in str(error)
    else:
        raise AssertionError("Mismatched decoder keys should fail")
