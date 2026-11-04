"""Reuse author data, seeding and evaluation; swap only the selection schedule."""
import train_author_gai as author
from src.author_random import RandomStrategy

author.entry.MyStrategy = RandomStrategy

if __name__ == '__main__':
    author.main()
