from typing import Literal
from pydantic import BaseModel, Field, confloat

#########################################################################################################################
# 4 Choices
#########################################################################################################################

# 1.1
class Response4Uppercase(BaseModel):
    thought: str = Field(
        description="A long-form description of chain-of-thought reasoning.",
    )

    answer: Literal["A", "B", "C", "D"] = Field(
        description="The most likely correct option.",
    )

    confidence: confloat(strict=True, ge=0.0, le=1.0) = Field(
        description="The confidence in the prediction as a float where 0.0 equals no confidence and 1.0 equals total confidence.",
    )

# 1.2
class Response4Lowercase(BaseModel):
    thought: str = Field(
        description="A long-form description of chain-of-thought reasoning.",
    )

    answer: Literal["a", "b", "c", "d"] = Field(
        description="The most likely correct option.",
    )

    confidence: confloat(strict=True, ge=0.0, le=1.0) = Field(
        description="The confidence in the prediction as a float where 0.0 equals no confidence and 1.0 equals total confidence.",
    )


# 1.3
class Response4Numeric(BaseModel):
    thought: str = Field(
        description="A long-form description of chain-of-thought reasoning.",
    )

    answer: Literal["1", "2", "3", "4"] = Field(
        description="The most likely correct option.",
    )

    confidence: confloat(strict=True, ge=0.0, le=1.0) = Field(
        description="The confidence in the prediction as a float where 0.0 equals no confidence and 1.0 equals total confidence.",
    )


# 2.1
class ResponseRelativeConfidence4Uppercase(BaseModel):
    thought: str = Field(
        description="A long-form description of chain-of-thought reasoning.",
    )

    answer: Literal["A", "B", "C", "D"] = Field(
        description="The most likely correct option.",
    )

    conf_a: confloat(strict=True, ge=0.0, le=1.0) = Field(
        description="The probability of A being correct as a float where 0.0 equals no confidence and 1.0 equals total confidence.",
    )

    conf_b: confloat(strict=True, ge=0.0, le=1.0) = Field(
        description="The probability of B being correct as a float where 0.0 equals no confidence and 1.0 equals total confidence.",
    )

    conf_c: confloat(strict=True, ge=0.0, le=1.0) = Field(
        description="The probability of C being correct as a float where 0.0 equals no confidence and 1.0 equals total confidence.",
    )

    conf_d: confloat(strict=True, ge=0.0, le=1.0) = Field(
        description="The probability of D being correct as a float where 0.0 equals no confidence and 1.0 equals total confidence.",
    )


# 2.2
class ResponseRelativeConfidence4Lowercase(BaseModel):
    thought: str = Field(
        description="A long-form description of chain-of-thought reasoning.",
    )

    answer: Literal["a", "b", "c", "d"] = Field(
        description="The most likely correct option.",
    )

    conf_a: confloat(strict=True, ge=0.0, le=1.0) = Field(
        description="The probability of a being correct as a float where 0.0 equals no confidence and 1.0 equals total confidence.",
    )

    conf_b: confloat(strict=True, ge=0.0, le=1.0) = Field(
        description="The probability of b being correct as a float where 0.0 equals no confidence and 1.0 equals total confidence.",
    )

    conf_c: confloat(strict=True, ge=0.0, le=1.0) = Field(
        description="The probability of c being correct as a float where 0.0 equals no confidence and 1.0 equals total confidence.",
    )

    conf_d: confloat(strict=True, ge=0.0, le=1.0) = Field(
        description="The probability of d being correct as a float where 0.0 equals no confidence and 1.0 equals total confidence.",
    )

# 2.3
class ResponseRelativeConfidence4Numeric(BaseModel):
    thought: str = Field(
        description="A long-form description of chain-of-thought reasoning.",
    )

    answer: Literal["1", "2", "3", "4"] = Field(
        description="The most likely correct option.",
    )

    conf_a: confloat(strict=True, ge=0.0, le=1.0) = Field(
        description="The probability of 1 being correct as a float where 0.0 equals no confidence and 1.0 equals total confidence.",
    )

    conf_b: confloat(strict=True, ge=0.0, le=1.0) = Field(
        description="The probability of 2 being correct as a float where 0.0 equals no confidence and 1.0 equals total confidence.",
    )

    conf_c: confloat(strict=True, ge=0.0, le=1.0) = Field(
        description="The probability of 3 being correct as a float where 0.0 equals no confidence and 1.0 equals total confidence.",
    )

    conf_d: confloat(strict=True, ge=0.0, le=1.0) = Field(
        description="The probability of 4 being correct as a float where 0.0 equals no confidence and 1.0 equals total confidence.",
    )


#########################################################################################################################
# 5 choices
#########################################################################################################################

# 3.1
class Response5Uppercase(BaseModel):
    thought: str = Field(
        description="A long-form description of chain-of-thought reasoning.",
    )

    answer: Literal["A", "B", "C", "D", "E"] = Field(
        description="The most likely correct option.",
    )

    confidence: confloat(strict=True, ge=0.0, le=1.0) = Field(
        description="The confidence in the prediction as a float where 0.0 equals no confidence and 1.0 equals total confidence.",
    )

# 3.2
class Response5Lowercase(BaseModel):
    thought: str = Field(
        description="A long-form description of chain-of-thought reasoning.",
    )

    answer: Literal["a", "b", "c", "d", "e"] = Field(
        description="The most likely correct option.",
    )

    confidence: confloat(strict=True, ge=0.0, le=1.0) = Field(
        description="The confidence in the prediction as a float where 0.0 equals no confidence and 1.0 equals total confidence.",
    )

# 3.3
class Response5Numeric(BaseModel):
    thought: str = Field(
        description="A long-form description of chain-of-thought reasoning.",
    )

    answer: Literal["1", "2", "3", "4", "5"] = Field(
        description="The most likely correct option.",
    )

    confidence: confloat(strict=True, ge=0.0, le=1.0) = Field(
        description="The confidence in the prediction as a float where 0.0 equals no confidence and 1.0 equals total confidence.",
    )

# 4.1
class ResponseRelativeConfidence5Uppercase(BaseModel):
    thought: str = Field(
        description="A long-form description of chain-of-thought reasoning.",
    )

    answer: Literal["A", "B", "C", "D", "E"] = Field(
        description="The most likely correct option.",
    )

    conf_a: confloat(strict=True, ge=0.0, le=1.0) = Field(
        description="The probability of A being correct as a float where 0.0 equals no confidence and 1.0 equals total confidence.",
    )

    conf_b: confloat(strict=True, ge=0.0, le=1.0) = Field(
        description="The probability of B being correct as a float where 0.0 equals no confidence and 1.0 equals total confidence.",
    )

    conf_c: confloat(strict=True, ge=0.0, le=1.0) = Field(
        description="The probability of C being correct as a float where 0.0 equals no confidence and 1.0 equals total confidence.",
    )

    conf_d: confloat(strict=True, ge=0.0, le=1.0) = Field(
        description="The probability of D being correct as a float where 0.0 equals no confidence and 1.0 equals total confidence.",
    )

    conf_e: confloat(strict=True, ge=0.0, le=1.0) = Field(
        description="The probability of E being correct as a float where 0.0 equals no confidence and 1.0 equals total confidence.",
    )


# 4.2
class ResponseRelativeConfidence5Lowercase(BaseModel):
    thought: str = Field(
        description="A long-form description of chain-of-thought reasoning.",
    )

    answer: Literal["a", "b", "c", "d", "e"] = Field(
        description="The most likely correct option.",
    )

    conf_a: confloat(strict=True, ge=0.0, le=1.0) = Field(
        description="The probability of a being correct as a float where 0.0 equals no confidence and 1.0 equals total confidence.",
    )

    conf_b: confloat(strict=True, ge=0.0, le=1.0) = Field(
        description="The probability of b being correct as a float where 0.0 equals no confidence and 1.0 equals total confidence.",
    )

    conf_c: confloat(strict=True, ge=0.0, le=1.0) = Field(
        description="The probability of c being correct as a float where 0.0 equals no confidence and 1.0 equals total confidence.",
    )

    conf_d: confloat(strict=True, ge=0.0, le=1.0) = Field(
        description="The probability of d being correct as a float where 0.0 equals no confidence and 1.0 equals total confidence.",
    )

    conf_e: confloat(strict=True, ge=0.0, le=1.0) = Field(
        description="The probability of e being correct as a float where 0.0 equals no confidence and 1.0 equals total confidence.",
    )

# 4.3
class ResponseRelativeConfidence5Numeric(BaseModel):
    thought: str = Field(
        description="A long-form description of chain-of-thought reasoning.",
    )

    answer: Literal["1", "2", "3", "4", "5"] = Field(
        description="The most likely correct option.",
    )

    conf_a: confloat(strict=True, ge=0.0, le=1.0) = Field(
        description="The probability of 1 being correct as a float where 0.0 equals no confidence and 1.0 equals total confidence.",
    )

    conf_b: confloat(strict=True, ge=0.0, le=1.0) = Field(
        description="The probability of 2 being correct as a float where 0.0 equals no confidence and 1.0 equals total confidence.",
    )

    conf_c: confloat(strict=True, ge=0.0, le=1.0) = Field(
        description="The probability of 3 being correct as a float where 0.0 equals no confidence and 1.0 equals total confidence.",
    )

    conf_d: confloat(strict=True, ge=0.0, le=1.0) = Field(
        description="The probability of 4 being correct as a float where 0.0 equals no confidence and 1.0 equals total confidence.",
    )

    conf_e: confloat(strict=True, ge=0.0, le=1.0) = Field(
        description="The probability of 5 being correct as a float where 0.0 equals no confidence and 1.0 equals total confidence.",
    )
