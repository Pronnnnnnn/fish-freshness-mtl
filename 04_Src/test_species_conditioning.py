"""Checks SpeciesConditionedMTL before any training is spent on it.

The gradient test is the one that matters. If the freshness loss can reach
the species head, this stops being "D-EW plus a conditioning channel" and
becomes a differently trained model, at which point nothing can be
attributed to conditioning.

Builds backbones without pretrained weights: the tests are about shapes and
gradient flow, neither of which depends on the initial values.
"""
import torch
import torch.nn as nn

from models import MultiTaskModel, SpeciesConditionedMTL

BATCH = 4
NUM_SPECIES = 8
NUM_FRESHNESS = 3


def test_forward_runs_and_shapes_match() -> tuple[int, int]:
    model = SpeciesConditionedMTL(pretrained=False)
    model.eval()
    species_logits, freshness_logits = model(torch.randn(BATCH, 3, 224, 224))

    assert species_logits.shape == (BATCH, NUM_SPECIES), species_logits.shape
    assert freshness_logits.shape == (BATCH, NUM_FRESHNESS), freshness_logits.shape
    print(f"  species_logits   {tuple(species_logits.shape)}")
    print(f"  freshness_logits {tuple(freshness_logits.shape)}")
    return species_logits.shape[1], freshness_logits.shape[1]


def test_parameter_count_difference() -> int:
    """D-Cond should differ from D-EW by exactly the 8x3 conditioning weights."""
    conditioned = sum(p.numel() for p in SpeciesConditionedMTL(pretrained=False).parameters())
    baseline = sum(p.numel() for p in MultiTaskModel(pretrained=False).parameters())
    difference = conditioned - baseline

    expected = NUM_SPECIES * NUM_FRESHNESS
    assert difference == expected, f"expected +{expected}, got {difference:+d}"
    print(f"  D-EW   {baseline:,}")
    print(f"  D-Cond {conditioned:,}  ({difference:+d})")
    return difference


def test_freshness_loss_does_not_reach_species_head() -> None:
    """The load-bearing one: detach must block the freshness gradient."""
    model = SpeciesConditionedMTL(pretrained=False)
    model.train()

    _, freshness_logits = model(torch.randn(BATCH, 3, 224, 224))
    target = torch.randint(0, NUM_FRESHNESS, (BATCH,))
    nn.CrossEntropyLoss()(freshness_logits, target).backward()

    for name, param in model.species_head.named_parameters():
        grad = param.grad
        magnitude = 0.0 if grad is None else float(grad.abs().sum())
        state = "None" if grad is None else f"sum|grad| = {magnitude:.3e}"
        print(f"  species_head.{name}: {state}")
        assert grad is None or magnitude == 0.0, (
            f"freshness loss reached species_head.{name}; the detach is missing "
            "and the experiment would not be valid"
        )

    # The conditioning layer itself must still receive gradient, otherwise the
    # detach was applied too broadly and the channel trains nothing.
    assert model.freshness_fc.weight.grad is not None
    assert float(model.freshness_fc.weight.grad.abs().sum()) > 0
    print(f"  freshness_fc.weight: sum|grad| = "
          f"{float(model.freshness_fc.weight.grad.abs().sum()):.3e} (nonzero, as required)")


def main() -> None:
    torch.manual_seed(0)

    print("1/4 forward pass and output shapes")
    test_forward_runs_and_shapes_match()

    print("\n2/4 parameter count against D-EW")
    test_parameter_count_difference()

    print("\n3/4 gradient isolation of the species head")
    test_freshness_loss_does_not_reach_species_head()

    print("\n4/4 species head still trains from its own loss")
    model = SpeciesConditionedMTL(pretrained=False)
    model.train()
    species_logits, _ = model(torch.randn(BATCH, 3, 224, 224))
    nn.CrossEntropyLoss()(species_logits, torch.randint(0, NUM_SPECIES, (BATCH,))).backward()
    grad_sum = float(model.species_head.fc.weight.grad.abs().sum())
    assert grad_sum > 0, "species loss should still train the species head"
    print(f"  species_head.fc.weight: sum|grad| = {grad_sum:.3e} (nonzero, as required)")

    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
