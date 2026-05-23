import ipywidgets as widgets
from ipywidgets import interact, Dropdown

question1 = "Select one of the options given:"

solution1 = {
    'A': 'Not quite. While both functions are used to create empty arrays, np.zeros() is initialized with the value 0.',
    'B': 'Not quite. np.zeros() is initialized, and it gives an output of 0s.',
    'C': 'Not quite. Most often, np.empty() is faster since it is not initialized.',
    'D': 'True! np.empty() creates an array with uninitialized elements from available memory space and may be faster to execute.'
}

def mcq(question, solution):
    print(question)
    print("Please select the correct option:")

    answer_w = Dropdown(
        options=[('Select an option', None)] + [(k, k) for k in solution.keys()],
        value=None,
        layout=widgets.Layout(width='25%')
    )

    @interact(Answer=answer_w)
    def show_answer(Answer):
        if Answer is not None:
            print(solution[Answer])

mcq(question1, solution1)
