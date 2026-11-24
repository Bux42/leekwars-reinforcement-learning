/* Some linkers refuse a shared library without any source of its own. The content
 * comes from the static library. */
int leekrl_shared_version(void);

int leekrl_shared_version(void)
{
    return 1;
}
