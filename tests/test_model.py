from talgin.model import ModelConfig, TalginDecoderModel


def test_v4_parameter_count():
    model = TalginDecoderModel(ModelConfig())
    assert model.parameter_count() == 66_207_360
