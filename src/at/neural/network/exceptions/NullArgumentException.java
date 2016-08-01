package at.neural.network.exceptions;

/**
 * Is thrown if an argument was null, but shouldn't be.
 */
public class NullArgumentException extends IllegalArgumentException {

    /**
     * @param paramName the name of the parameter that was null.
     */
    public NullArgumentException(String paramName) {
        super("The parameter '" + paramName + "' must not be null!");
    }
}
