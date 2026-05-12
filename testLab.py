def z_transform(sequence):
    terms = []

    for n, val in enumerate(sequence):
        if val == 0:
            continue

        if n == 0:
            term = f"{val}"
        else:
            z_part = f"z^-{n}"
            if val == 1:
                term = z_part
            elif val == -1:
                term = f"-{z_part}"
            else:
                term = f"{val}{z_part}"

        terms.append(term)

    return " + ".join(terms).replace("+ -", "- ") if terms else "0"


while True:
    user_input = input("\nEnter sequence (or 'q' to quit): ")

    if user_input.lower() == 'q':
        print("Program ended.")
        break

    try:
        sequence = [int(x) for x in user_input.split()]

        if not sequence:
            print("Please enter AT LEAST ONE number.")
            continue

        result = z_transform(sequence)
        print(f"X(z) = {result}")

    except ValueError:
        print("Invalid input. NUMBERS ONLY.")