.. _development-reference:

Development
-----------

Notes for developers. If you want to get involved, please do!

Releasing
+++++++++

Releasing is semi-automated. The steps required are the following:

#. Bump the version. This is done via the "Bump version" action, which must be triggered manually.
#. Once this action has completed, it will then create a draft release on GitHub (which will appear at `<https://github.com/openscm/OpenSCM-Calibration/releases>`_)
#. Review this release, update any announcements and check the changes, then publish
#. Upon publication, the release is automatically uploaded to PyPI via the "Release" action
#. That's it, release done, make noise on social media of choice, do whatever else
#. Enjoy the newly available version
