def main():

    receptive_field = 1
    jump = 1

    def conv3(rf, j):
        return rf + 2 * j

    def pool2(rf, j):
        rf = rf + j
        j *= 2
        return rf, j

    print(
        f"Input: RF={receptive_field}, jump={jump}"
    )

    for level in range(4):

        receptive_field = conv3(
            receptive_field,
            jump
        )

        receptive_field = conv3(
            receptive_field,
            jump
        )

        print(
            f"Encoder {level + 1}: "
            f"RF={receptive_field}"
        )

        receptive_field, jump = pool2(
            receptive_field,
            jump
        )

    receptive_field = conv3(
        receptive_field,
        jump
    )

    receptive_field = conv3(
        receptive_field,
        jump
    )

    print(
        f"Bottleneck: RF={receptive_field}"
    )

    for level in range(4):

        jump //= 2

        receptive_field = conv3(
            receptive_field,
            jump
        )

        receptive_field = conv3(
            receptive_field,
            jump
        )

        print(
            f"Decoder {level + 1}: "
            f"RF={receptive_field}"
        )

    print()
    print(
        "Maximum theoretical receptive field:",
        receptive_field
    )


if __name__ == "__main__":
    main()
