"""
    mail@kaiploeger.net
"""

import click               # for command line arguments
import InquirerPy as iq    # for interactive menus


# Click is similar to argparse, but is more aimed to write whole applications with different commands
# and takes care of the control flow while arparse intends to "pass arguments to a program".


@click.group()
def app():
   """Welcome to this test of click and InquirerPy!"""


@app.command(name='hello')
@click.argument('name', type=click.STRING)
@click.option('--greeting', '-g', type=click.STRING, default='Hello')
def run(name, greeting):
    """Say hello to NAME"""
    click.echo(f'{greeting} {name}!')


@app.command(name='ask')
def this_does_not_need_to_be_named_ask():
    """Ask a question"""

    feeling_prompt = {
        'type': 'list',
        'name': 'feeling',
        'message': 'How are you?',
        'choices': ['good', 'meh', 'bad'],
    }
    answer_feeling = iq.prompt(feeling_prompt, vi_mode=True)

    food_prompt = {
        'type': 'checkbox',
        'name': 'food',
        'message': 'What do you want for breakfast?',
        'choices': ['eggs', 'french toast', 'orange juice', 'pancackes'],
    }

    answer_food = iq.prompt(food_prompt, vi_mode=True)
    answers = answer_feeling | answer_food # merge dictionaries

    click.echo(f'You are {answers["feeling"]} and would like to eat {answers["food"]}')



if __name__ == '__main__':
    app()
