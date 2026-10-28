import unittest
from unittest.mock import MagicMock

import main


class TestMultiAdmin(unittest.TestCase):
    """Verifies that ADMIN_USER_ID supports multiple comma- or space-separated Discord user IDs."""

    def setUp(self) -> None:
        self.orig_admin = main.ADMIN_USER_ID
        self.orig_admin_set = main._admin_user_ids

    def tearDown(self) -> None:
        main.ADMIN_USER_ID = self.orig_admin
        main._admin_user_ids = self.orig_admin_set

    def test_single_admin(self) -> None:
        main.set_app_state(admin_id="123456789")
        self.assertTrue(main.is_admin_user("123456789"))
        self.assertFalse(main.is_admin_user("999999999"))

    def test_comma_separated_admins(self) -> None:
        main.set_app_state(admin_id="111111, 222222, 333333")
        self.assertTrue(main.is_admin_user("111111"))
        self.assertTrue(main.is_admin_user("222222"))
        self.assertTrue(main.is_admin_user("333333"))
        self.assertFalse(main.is_admin_user("444444"))

    def test_space_and_semicolon_separated_admins(self) -> None:
        main.set_app_state(admin_id="111111 222222;333333")
        self.assertTrue(main.is_admin_user("111111"))
        self.assertTrue(main.is_admin_user("222222"))
        self.assertTrue(main.is_admin_user("333333"))
        self.assertFalse(main.is_admin_user("555555"))

    def test_interaction_is_admin_for_multiple_admins(self) -> None:
        main.set_app_state(admin_id="111,222")

        user1 = MagicMock()
        user1.id = 111
        interaction1 = MagicMock()
        interaction1.user = user1

        user2 = MagicMock()
        user2.id = 222
        interaction2 = MagicMock()
        interaction2.user = user2

        user_other = MagicMock()
        user_other.id = 999
        interaction_other = MagicMock()
        interaction_other.user = user_other

        self.assertTrue(main.is_admin(interaction1))
        self.assertTrue(main.is_admin(interaction2))
        self.assertFalse(main.is_admin(interaction_other))


if __name__ == "__main__":
    unittest.main()
