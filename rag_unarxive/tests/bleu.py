from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

reference = [
    'bb this is a dog'.split(),
    'bb it is dog'.split(),
    'bb dog it is'.split(),
    'bb a dog, it is'.split()
]
print(reference)

candidate = 'bb it is dog'.split()
print('BLEU score -> {}'.format(sentence_bleu(reference, candidate))) # should be 1, not 0

candidate = 'it is a dog'.split()
print('Individual 1-gram: %f' % sentence_bleu(reference, candidate, weights=(1, 0, 0, 0)))
print('Individual 2-gram: %f' % sentence_bleu(reference, candidate, weights=(0, 1, 0, 0)))
print('Individual 3-gram: %f' % sentence_bleu(reference, candidate, weights=(0, 0, 1, 0)))
print('Individual 4-gram: %f' % sentence_bleu(reference, candidate, weights=(0, 0, 0, 1)))

score = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))
print(score)


# Natural Language Toolkit: BLEU Score
"""BLEU score implementation."""
hypothesis1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'which',
              'ensures', 'that', 'the', 'military', 'always',
              'obeys', 'the', 'commands', 'of', 'the', 'party']

hypothesis2 = ['It', 'is', 'to', 'insure', 'the', 'troops',
              'forever', 'hearing', 'the', 'activity', 'guidebook',
              'that', 'party', 'direct']

reference1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'that',
              'ensures', 'that', 'the', 'military', 'will', 'forever',
              'heed', 'Party', 'commands']

reference2 = ['It', 'is', 'the', 'guiding', 'principle', 'which',
              'guarantees', 'the', 'military', 'forces', 'always',
              'being', 'under', 'the', 'command', 'of', 'the',
              'Party']

reference3 = ['It', 'is', 'the', 'practical', 'guide', 'for', 'the',
              'army', 'always', 'to', 'heed', 'the', 'directions',
              'of', 'the', 'party']

print(sentence_bleu([reference1, reference2, reference3], hypothesis1))
print(round(sentence_bleu([reference1, reference2, reference3], hypothesis2),4))

chencherry = SmoothingFunction()
print(sentence_bleu([reference1, reference2, reference3], hypothesis2, smoothing_function=chencherry.method1))

weights = (1./5., 1./5., 1./5., 1./5., 1./5.)
print(sentence_bleu([reference1, reference2, reference3], hypothesis1, weights)) 

weights = [
    (1./2., 1./2.),
    (1./3., 1./3., 1./3.),
    (1./4., 1./4., 1./4., 1./4.)
]
print(sentence_bleu([reference1, reference2, reference3], hypothesis1, weights)) 
