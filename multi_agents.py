import numpy as np
import abc
import util
from game import Agent, Action


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def get_action(self, game_state):
        """
        You do not need to change this method, but you're welcome to.

        get_action chooses among the best options according to the evaluation function.

        get_action takes a game_state and returns some Action.X for some X in the set
        {UP, DOWN, LEFT, RIGHT, STOP}
        """

        # Collect legal moves and successor states
        legal_moves = game_state.get_agent_legal_actions()

        # Choose one of the best actions
        scores = [self.evaluation_function(game_state, action) for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = np.random.choice(best_indices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legal_moves[chosen_index]

    def evaluation_function(self, current_game_state, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (GameState.py) and returns a number, where higher numbers are better.

        """

        # Useful information you can extract from a GameState (game_state.py)

        successor_game_state = current_game_state.generate_successor(action=action)
        board = successor_game_state.board
        max_tile = successor_game_state.max_tile
        score = successor_game_state.score

        "*** YOUR CODE HERE ***"
        weight = [[1, 2, 4, 8], [8, 16, 32, 64], [64, 128, 256, 512], [512, 1024, 2048, 4069]]

        successor_sum = 0
        for i in range(len(board)):
            for j in range(len(board[i])):
                successor_sum += board[i][j] * weight[i][j]

        return successor_sum


def score_evaluation_function(current_game_state):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return current_game_state.score


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinmaxAgent, AlphaBetaAgent & ExpectimaxAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evaluation_function='scoreEvaluationFunction', depth=2):
        self.evaluation_function = util.lookup(evaluation_function, globals())
        self.depth = depth

    @abc.abstractmethod
    def get_action(self, game_state):
        return


class MinmaxAgent(MultiAgentSearchAgent):
    def get_action(self, game_state):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        game_state.get_legal_actions(agent_index):
            Returns a list of legal actions for an agent
            agent_index=0 means our agent, the opponent is agent_index=1

        Action.STOP:
            The stop direction, which is always legal

        game_state.generate_successor(agent_index, action):
            Returns the successor game state after an agent takes an action

        ***************************************************************************
        The Minimax pseudocode we relied on:
        Unit 3 - page 90 - Minimax Algorithm
        https://en.wikipedia.org/wiki/Minimax
        https://athena.ecs.csus.edu/~gordonvs/Beijing/Minimax.pdf
        ***************************************************************************
        """

        """*** YOUR CODE HERE ***"""

        best_value = -np.inf
        best_move = None
        for move in game_state.get_legal_actions(0):
            value = self.minimizing(game_state.generate_successor(0, move), self.depth)
            if value > best_value:
                best_value = value
                best_move = move
        return best_move

    def maximizing(self, game_state, depth):  # agent 0
        """
        This is our maximizer. It will try to get the highest score possible
        :param game_state: Our game state
        :param depth: The depth we are at
        :return: The maximum score
        """
        if not game_state.get_legal_actions(0):  # legal actions  = []
            return self.evaluation_function(game_state)  # someone won or draw
        if depth == 0:
            return self.evaluation_function(game_state)

        value = -np.inf
        for move in game_state.get_legal_actions(0):
            value = max(value, self.minimizing(game_state.generate_successor(0, move),
                                               depth - 1))
        return value

    def minimizing(self, game_state, depth):  # agent 1
        """
        This is our minimizer. It will try to get the lowest score possible.
        :param game_state: Our game state
        :param depth: The depth we are at
        :return: The minimum score
        """
        if not game_state.get_legal_actions(0):  # legal actions = []
            return self.evaluation_function(game_state)  # someone won or draw
        if depth == 0:
            return self.evaluation_function(game_state)

        value = np.inf
        for move in game_state.get_legal_actions(1):
            value = min(value, self.maximizing(game_state.generate_successor(1, move),
                                               depth - 1))
        return value


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def get_action(self, game_state):
        """
        Returns the minimax action using self.depth and self.evaluationFunction

        ***************************************************************************
        The alpha-beta pruning algorithm: seeks to decrease the number of nodes being
        evaluated by the minimax algorithm in the search tree. It stops evaluating a
        move when at least one possibility has been found that proves to be worse than
        a previously examined move.

        The alpha-beta pruning pseudocode we relied on:
        Unit 3 - page 96 - The \alpha-\beta Algorithm
        https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning
        ***************************************************************************
        """

        """*** YOUR CODE HERE ***"""

        best_value = -np.inf
        best_move = None
        for move in game_state.get_legal_actions(0):
            value = self.minimizing(game_state.generate_successor(0, move), self.depth,
                                    -np.inf, np.inf)
            if value > best_value:
                best_value = value
                best_move = move
        return best_move

    def maximizing(self, game_state, depth, alpha, beta):  # agent 0
        """
        This is our maximizer. It will try to get the highest score possible, using the
        alpha-beta pruning algorithm.
        :param game_state: Our game state
        :param depth: The depth we are at
        :param alpha: the highest possible score found so far - initial value: -infinity
        :param beta: the lowest possible score found so far - initial value: infinity
        :return: The maximum score
        """
        if not game_state.get_legal_actions(0):  # legal actions  = []
            return self.evaluation_function(game_state)  # someone won or draw
        if depth == 0:
            return self.evaluation_function(game_state)

        value = -np.inf
        for move in game_state.get_legal_actions(0):
            value = max(value, self.minimizing(game_state.generate_successor(0, move),
                                               depth - 1, alpha, beta))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value

    def minimizing(self, game_state, depth, alpha, beta):  # agent 1
        """
        This is our minimizer. It will try to get the lowest score possible, using the
        alpha-beta pruning algorithm.
        :param game_state: Our game state
        :param depth: The depth we are at
        :param alpha: the highest possible score found so far - initial value: -infinity
        :param beta: the lowest possible score found so far - initial value: infinity
        :return: The maximum score
        """
        if not game_state.get_legal_actions(0):  # legal actions  = []
            return self.evaluation_function(game_state)  # someone won or draw
        if depth == 0:
            return self.evaluation_function(game_state)

        value = np.inf
        for move in game_state.get_legal_actions(1):
            value = min(value, self.maximizing(game_state.generate_successor(1, move),
                                               depth - 1, alpha, beta))
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def get_action(self, game_state):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        The opponent should be modeled as choosing uniformly at random from their
        legal moves.

        ***************************************************************************
        The expectimax algorithm: maximizes the expected utility. This algorithm doesn't
        assume that the minimizing player plays optimally, unlike the minimax algorithm.
        Here the minimizing players moves and actions are based on chance.
        Thus, We replace the minimizing nodes by chance nodes, as there is the
        possibility of the minimizer not playing optimally or making a
        mistake. Chance nodes take the average of all children and returns
        the expected value.
        This helps take advantage of non-optimal opponents and can take
        risks to end up in a state of higher utility since the opponent is
        random.

        The Expectimax pseudocode we relied on:
        https://en.wikipedia.org/wiki/Expectiminimax
        https://www.geeksforgeeks.org/expectimax-algorithm-in-game-theory/
        ***************************************************************************
        """

        """*** YOUR CODE HERE ***"""

        best_value = -np.inf
        best_move = None
        for move in game_state.get_legal_actions(0):
            value = self.minimizing(game_state.generate_successor(0, move), self.depth)
            if value > best_value:
                best_value = value
                best_move = move
        return best_move

    def maximizing(self, game_state, depth):  # agent 0
        """
        This is our maximizer. It will try to get the lowest score possible, using the
        expectimax algorithm.
        :param game_state: Our game state
        :param depth: The depth we're at
        :return: The maximum score
        """
        if not game_state.get_legal_actions(0):  # legal actions  = []
            return self.evaluation_function(game_state)  # someone won or draw
        if depth == 0:
            return self.evaluation_function(game_state)

        value = -np.inf
        for move in game_state.get_legal_actions(0):
            value = max(value, self.minimizing(game_state.generate_successor(0, move),
                                               depth - 1))
        return value

    def minimizing(self, game_state, depth):  # agent 1
        """
        This is our minimizer. It will try to get the lowest score possible, using the
        expectimax algorithm.
        :param game_state: Our game state
        :param depth: The depth we're at
        :return: The minimum score
        """
        if not game_state.get_legal_actions(0):  # legal actions  = []
            return self.evaluation_function(game_state)  # someone won or draw
        if depth == 0:
            return self.evaluation_function(game_state)

        successors_sum = 0
        successors_count = 0
        for move in game_state.get_legal_actions(1):
            value = self.maximizing(game_state.generate_successor(1, move), depth - 1)
            successors_sum += value
            successors_count += 1

        successors_average = successors_sum / successors_count
        return successors_average


def better_evaluation_function(current_game_state):
    """
    Your extreme 2048 evaluation function (question 5).

    DESCRIPTION:

    Funny story, I am addicted to this game and have been playing it since it came out.
    I usually like to shift my highest valued square into the bottom right corner and
    keep on adding to it.

    *Best way to not lose (I've learned) is to begin by shifting the blocks to the bottom
    rows and into the right bottom corner and keep on filling the bottom row (move the
    tiles right and down only and left if right and down aren't possible, never up).
    That way, we fill the bottom row with the highest valued tiles where the largest valued
    tiles is the utmost right one and the second highest valued is the second to right,
    and so on.

    Knowing we need to shift our highest value into the bottom right corner and the second
    highest into the one left of it and so forth, we knew we needed to show that the bottom
    row is of highest weight (and largest ascent).

    regarding the weight of each tile, we chose to work our way up from 1 to 4096:
    [1,2,4,8,16,32,64,128,256,512,1024,2048,4069]

    Now, we'll check all the possible successor states of the current state and calculate
    which is of largest weight by summing up each tile multiplied by its weight, and
    return the highest sum. The highest summed action is the best action to take.

    While researching, we found a few amazing articles that cover this topic:
    http://cs229.stanford.edu/proj2016/report/NieHouAn-AIPlays2048-report.pdf

    http://www.randalolson.com/2015/04/27/artificial-intelligence-has-crushed-all-human-records-in-2048-heres-how
    -the-ai-pulled-it-off/
    """
    "*** YOUR CODE HERE ***"

    weight = [[1, 2, 4, 8], [8, 16, 32, 64], [64, 128, 256, 512], [512, 1024, 2048, 4069]]

    if not current_game_state.get_legal_actions(0):  # legal actions  = []
        return current_game_state.score  # someone won or draw

    best_score = 0
    for action in current_game_state.get_legal_actions(0):
        successor_game_state = current_game_state.generate_successor(action=action)

        successor_sum = 0
        for i in range(len(successor_game_state.board)):
            for j in range(len(successor_game_state.board[i])):
                successor_sum += successor_game_state.board[i][j] * weight[i][j]
        if successor_sum > best_score:
            best_score = successor_sum

    return best_score


# Abbreviation
better = better_evaluation_function
