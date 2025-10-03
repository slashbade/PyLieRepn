import unittest
import numpy as np
from LieToolbox.Repn.GK_dimension import GK_dimension
from LieToolbox.Repn.utils import clear_images

class TestGKDimension(unittest.TestCase):
    def setUp(self) -> None:
        clear_images()
    
    def tearDown(self) -> None:
        clear_images()

    def testF41(self) -> None:
        gk_dim, info = GK_dimension('F', 4, np.array([4, 5, 3/2, 1/2]))
        self.assertEqual(eval(gk_dim), 15)
        self.assertEqual(info['dual'], 'A_2')

    def testF42(self) -> None:
        gk_dim, info = GK_dimension('F', 4, np.array([2, 4, 5, 7]))
        self.assertEqual(eval(gk_dim), 21)
        self.assertEqual(info['dual'], 'B_3')

    def testF43(self) -> None:
        gk_dim, info = GK_dimension('F', 4, np.array([2, 4, 1, 7]))
        self.assertEqual(eval(gk_dim), 14)
        self.assertEqual(info['dual'], 'A_1 + \\tilde{A}_1')
    
    def testF44(self) -> None:
        gk_dim, info = GK_dimension('F', 4, np.array([7, 1, 5, -3])/4)
        self.assertEqual(eval(gk_dim), 19)
        self.assertEqual(info['dual'], 'C_3(a_1)')

    def testE61(self) -> None:
        gk_dim, info = GK_dimension('E', 6, np.array([1, 2, 1, 4, 4.5, 0.5, 0.5, -0.5]))
        self.assertEqual(eval(gk_dim), 29)
        self.assertEqual(info['dual'], 'D_4(a_1)')

    def testF71(self) -> None:
        gk_dim, info = GK_dimension('E', 7, np.array([1, 1, 1, 1, 1, -3, -4, 4])/4)
        self.assertEqual(eval(gk_dim), 60)

    def testE72(self) -> None:
        gk_dim, info = GK_dimension('E', 7, np.array([1, 3, -5, -7, -9, -11, -1/2, 1/2]))
        self.assertEqual(eval(gk_dim), 60)
        self.assertEqual(info['dual'], 'E_6')

    def testE73(self) -> None:
        gk_dim, info = GK_dimension('E', 7, np.array([-1, 1, 1, 1, 1, 1, -5, 5])/4)
        self.assertEqual(eval(gk_dim), 59)

    def testE74(self) -> None:
        gk_dim, info = GK_dimension('E', 7, np.array([0, 2, 2, 4, 6, 6, -14, 14])/8)
        self.assertEqual(eval(gk_dim), 55)
        self.assertEqual(info['dual'], 'D_6(a_2)')

    def testE81(self) -> None:
        gk_dim, info = GK_dimension('E', 8, np.array([1, 5, 9, 13, 9, 1, 5, 9])/4)
        self.assertEqual(eval(gk_dim), 103)

    def testE82(self) -> None:
        gk_dim, info = GK_dimension('E', 8, np.array([1/2, -3/2, -3, -2, -1, -4, -5, -19]))
        self.assertEqual(eval(gk_dim), 117)

    def testE83(self) -> None:
        gk_dim, info = GK_dimension('E', 8, np.array([1, 1, 1, 1, 1, 1, 1/2, 5/2]))
        self.assertEqual(eval(gk_dim), 113)
        self.assertEqual(info['dual'], 'D_7')

    def testE84(self) -> None:
        gk_dim, info = GK_dimension('E', 8, np.array([0, 0, 1/3, 1/3, 1/3, 1/3, 2/3, 6/3]))
        self.assertEqual(eval(gk_dim), 113)
        self.assertEqual(info['dual'], 'E_8(b_5)')

    def testE85(self) -> None:
        gk_dim, info = GK_dimension('E', 8, np.array([0, 0, 1/4, 1/4, 2/4, 4/4, 6/4, 16/4]))
        self.assertEqual(eval(gk_dim), 104)
        self.assertEqual(info['dual'], 'A_1 + D_5')
    
    def testE86(self) -> None:
        gk_dim, info = GK_dimension('E', 8, np.array([1/8, 1/8, 1/8, 1/8, 1/8, 3/8, 7/8, 13/8]))
        self.assertEqual(info['dual'], 'E_7')
    
    def testE87(self) -> None:
        gk_dim, info = GK_dimension('E', 8, np.array([1, 3, -5, -7, -9, -11, -1/2, 1/2]))
        self.assertEqual(eval(gk_dim), 113)
        self.assertEqual(info['dual'], 'E_8(b_5)')
    
    def testE88(self) -> None:
        gk_dim, info = GK_dimension('E', 8, np.array([0, 0, 0, 0, 0, 1/2, 1, 3/2]))
        self.assertEqual(eval(gk_dim), 116)
        self.assertEqual(info['dual'], 'E_7')
    
    def testE89(self) -> None:
        gk_dim, info = GK_dimension('E', 8, np.array([0, 0, 0, 0, 0, 1/2, 1, 5/2]))
        self.assertEqual(eval(gk_dim), 112)
        self.assertEqual(info['dual'], 'E_7(a_2)')
   


