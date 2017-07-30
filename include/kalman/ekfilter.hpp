// This file is part of kfilter.
// kfilter is a C++ variable-dimension extended kalman filter library.
//
// The comments have been updated in this code by:
//   Joe Montgomery 2017
// however the implementation is taken from:
//
// Copyright (C) 2004        Vincent Zalzal, Sylvain Marleau
// Copyright (C) 2001, 2004  Richard Gourdeau
// Copyright (C) 2004        GRPR and DGE's Automation sector
//                           �cole Polytechnique de Montr�al
//
// Code adapted from algorithms presented in :
//      Bierman, G. J. "Factorization Methods for Discrete Sequential
//      Estimation", Academic Press, 1977.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

#ifndef EKFILTER_HPP
#define EKFILTER_HPP

/*!
 \file
\brief Contains the interface of the \c EKFilter base template class.
*/

#include "kalman/kvector.hpp"
#include "kalman/kmatrix.hpp"

namespace Kalman {

  /*!
   \anchor templateEKFilter
   Generic Extended %Kalman Filter (EKF) template base class.

    The user of this class must derive from it, and implement all the 
    functions corresponding to \a A, \a W, \a Q, f, \a H, \a V, \a R
    and h.
     
    - makeProcess() corresponds to the non-linear prediction function:
    \f[ \tilde{x}_k = f \left( \hat{x}_{k-1}, u_{k-1}, w_{k-1} \right) \f]
    - makeMeasure() corresponds to the non-linear measurement function:
    \f[ \tilde{z}_k = h \left( \tilde{x}_k, v_k \right) \f]

    - makeA() , makeW() corresponds to the linearized process equation:
    \f$ x_k = \tilde{x}_k + A \left( x_{k-1} - \hat{x}_{k-1} \right) + W w_{k-1} \f$

    - makeH() and makeV() corresponds to the linearized measurement equation
    \f$ z_k = \tilde{z}_k + H \left( x_k - \tilde{x}_k \right) + V v_k \f$

    While makeQ() and makeR() correspond to the covariance matrices Q and R in:
    - \f$ Q = E \left( w w^T \right) \f$ describing the process noise
    - \f$ R = E \left( v v^T \right) \f$ describing the meansurement noise

    where
    - A is the jacobian matrix of \f$ f(x,u_{k-1},0) \f$.
    - W is the jacobian matrix of \a \f$ f(\hat{x}_{k-1},w) \f$
    - Q is the process noise covariance matrix.
    - H is the jacobian matrix of \f$ h(x, 0) \f$.
    - V is the jacobian matrix of \f$ h(\hat{x}_{k-1}, v)\f$.
    - R is the measurement noise covariance matrix.

    \par Template parameters
    - \c T : Type of elements contained in matrices and vectors. Usually 
             \c float or \c double.
    - \c BEG : Starting index of matrices and vectors. Can be either 0 or 1.
    - \c OQ : Optimize calculations on \a Q. This can be turned on if \a Q 
              is diagonal.
    - \c OVR : Optimize calculations on \a V and \a R. This can be turned on
               if \a V and \a R are both diagonal matrices.
    - \c DGB : Debug flag. If \c true, then bound-checking will be performed,
               and \c OutOfBoundError exceptions can be thrown.
     
    \par Type requirements for T
    - \c T must be <b>default constructible</b>.
    - \c T must be <b>constructible from</b> \c double.
    - \c T must be \b assignable.
    - \c T must be <b>equality comparable</b>.
    - \c T must be \b serializable.
    - \c T must support <b>basic arithmetic operations</b>.
    .
    This means that, if \c t1, \c t2 are instances of \c T, 
    \c op is an arithmetic operator (+ - * /),
    \c is is of type
    \c istream and \c os is of type \c ostream, the following
    expressions must be valid :
    - \code T(); T t1; \endcode Default constructor
    - \code T(0.0); T t1(1.0); \endcode Constructor from \c double
    - \code T t1 = t2; T t1(t2); T(t1); \endcode Copy constructor
    - \code t1 op t2 \endcode Arithmetic operation, convertible to \c T
    - \code -t1 \endcode Negation operator, convertible to \c T.
          Same as : \code T(0.0) - t1; \endcode
    - \code t1 = t2; \endcode Assignment operator
    - \code t1 op= t2; \endcode Arithmetic inplace operation.
          Same as : \code t1 = t1 op t2; \endcode
    - \code t1 == t2 \endcode Equality comparison, convertible to \c bool
    - \code is >> t1; \endcode \c operator>>()
    - \code os << t1; \endcode \c operator<<()
      
    Finally, note that \c operator>>() and \c operator<<() must be
    compatible. Also, \c operator&() must not have been overloaded.
     
  */
  template<typename T, K_UINT_32 BEG, bool OQ = false, 
           bool OVR = false, bool DBG = true>
  class EKFilter {
  public:

    typedef T type;     //!< Type of objects contained in matrices and vectors.

    enum { beg = BEG    //!< Starting index of matrices and vectors.
    };

    typedef KVector<T, BEG, DBG> Vector;  //!< Vector type.
    typedef KMatrix<T, BEG, DBG> Matrix;  //!< Matrix type.

    //! \name Constructor and Destructor.
    //!  @{

    EKFilter(); //!< Default constructor
    
    //! Constructor specifying all necessary matrix and vector dimensions.
    //! This constructor simply calls \c setDim() with
    //! \c n_ = dimension of \b x the state vector, \c nu_= dimension of \b u the
    //! control input vector, \c nw_= dimension of \b w the process noise
    //! \c m_ = dimension of \b z the measurement vector and \c nv_= dimension of
    //! \b v the measurement noise.
    EKFilter(K_UINT_32 n_, K_UINT_32 nu_, K_UINT_32 nw_, 
             K_UINT_32 m_, K_UINT_32 nv_); //!< sets the initial dimensions see setDim()

    //! Virtual destructor.
    virtual ~EKFilter();

    //! @}

    //! \name Dimension Accessor Functions
    //! @{

    //! Returns the size of the state vector.
    K_UINT_32 getSizeX() const;

    //! Returns the size of the input vector.
    K_UINT_32 getSizeU() const;

    //! Returns the size of the process noise vector.
    K_UINT_32 getSizeW() const;

    //! Returns the size of the measurement vector.
    K_UINT_32 getSizeZ() const;

    //! Returns the size of the measurement noise vector.
    K_UINT_32 getSizeV() const;

    //! @}

    /*! \name Resizing Functions
        These functions change the dimensions of all matrices and
        vectors supporting implementing a Variable-Dimension 
        Extended %Kalman Filter.
        They do nothing if the new size is the same as the old one.
        \warning \c setDim() (or the five \c setSize functions) \b must be 
        called \b before any other function, or else, matrices and vectors 
        will not have their memory allocated.
    */

    //! @{
    // TODO !!! watch out : i don't know which dims can be 0 !

    /*!
        Sets all dimensions at once.
        This function simply calls the \c setSize*() functions with
        \c n_ = dimension of \b x the state vector, \c nu_= dimension of \b u the
        control input vector, \c nw_= dimension of \b w the process noise
        \c m_ = dimension of \b z the measurement vector and \c nv_= dimension of
        \b v the measurement noise.

        \warning This function (or the corresponding five \c setSize*() 
        functions) must be called before any other functions.
        \warning \c init() must always be called after this function and
        before any other non-dimensioning function.
    */
    void setDim(K_UINT_32 n_, K_UINT_32 nu_, K_UINT_32 nw_, 
                K_UINT_32 m_, K_UINT_32 nv_);

    /*! Sets the size of the state vector.
        \param n_ New state vector size. Must not be 0.
        \warning \c init() must always be called after this function and
        before any other non-dimensioning function.
    */
    void setSizeX(K_UINT_32 n_);

    //! Sets the size of the input vector.
    void setSizeU(K_UINT_32 nu_);

    /*! Sets the size of the process noise vector.
        \param nw_ New process noise vector size.
        \warning \c init() must always be called after this function and
        before any other non-dimensioning function.
    */
    void setSizeW(K_UINT_32 nw_);

    //! Sets the size of the measurement vector.
    void setSizeZ(K_UINT_32 m_);

    //! Sets the size of the measurement noise vector.
    void setSizeV(K_UINT_32 nv_);

    //! @}

    /*! Sets initial conditions for the %Kalman Filter.
        This function sets an initial state estimate vector and an
        initial error covariance matrix estimate. This must be called at least
        once, after all dimensioning functions and before any other function.
        However, it can also be called anytime to reset or modify \a x or
        \a P.
        \param x_ State vector estimate. Will be destroyed.
        \param P_ Error covariance matrix estimate. Will be destroyed.
        \warning If \c setDim(), \c setSizeX() or \c setSizeW() is called,
        then init() must be called again before any other non-dimensioning
        function.
    */
    void init(Vector& x_, Matrix& P_);

    /*! \name %Kalman Filter Functions
        These functions get the results from the %Kalman filtering
        algorithm. Before any of these can be called, all dimensions must have
        been set properly at least once and \c init() must have been called,
        also at least once. Each time the user want to resize some vectors,
        the corresponding resizing functions must be called again before
        being able to call one of the functions in this section. \c init()
        must also be called again if \a n or \a nw has changed. \c init()
        can also be called solely to reset the filter.
    */
    //! @{

    /*! Performs one prediction-correction step. This will call
        the prediction \c timeUpdateStep() method that implements
        the prediction equation.

        Next it calls the correction \c measureUpdateStep()
        method that implements the measurement update equation.

        \param u_ Input vector. Will \b not be destroyed. Can be empty.
        \param z_ Measurement vector. Will \b not be destroyed. Can be empty.
    */
    void step(Vector& u_, const Vector& z_);

    /*! Performs one prediction step.
        The prediction equations are:
        - \f$ x_k = \tilde{x}_k + A \left( x_{k-1} - \hat{x}_{k-1} \right) + W w_{k-1} \f$
        - \f$ Q = E \left( w w^T \right) \f$

        This is the main \c EKFilter function. First, it resizes any matrix
        that needs it. Then, it proceeds to the time update phase, using
        the input vector \c u_. This means that the following virtual 
        functions <em>will be</em> called :
        - \c makeCommonProcess() - called on each step to do any common
        calculations shared by the \c makeA(), \c makeW(),
        \c makeQ() and \c makeProcess() functions.
        - \c makeA() creates the linearized A matrix
        - \c makeW() creates the 
        - \c makeQ() 
        - \c makeProcess()

        Upon return of the function, \a x contains the current predicted state
        \f$ \tilde{x}_k \f$ instead of the previous iteration's corrected state
        \f$ \hat{x}_{k-1} \f$. 

        \param u_ Input vector. Will \b not be destroyed. Can be empty.
    */
    void timeUpdateStep(Vector& u_);

    /*! Performs one correction step.
	The measurement/correction equations are:

        - \f$ z_k = \tilde{z}_k + H \left( x_k - \tilde{x}_k \right) + V v_k \f$
        - \f$ R = E \left( v v^T \right) \f$

        If \c z_ is empty, that is, if there are no measurements in this 
        step, there is no correction and the function stops there. 
        Otherwise, the correction/measurement update phase begins.

        First, this calls the member function \c sizeUpdate() so that
        \c sizeUpdate() can resize any matrix that needs it.

        Next, this calls the following virtual member functions to set up
        the matrices:
	- \c makeCommonMeasure() 
        - \c makeHImpl()
	- \c makeVImpl()
	- \c makeRImpl()
	- \c makeMeasure()
        - \c makeDZ()

	After this correction phase ends, \a x contains the new corrected state
        \f$ \hat{x}_k \f$.

        \param z_ Measurement vector. Will \b not be destroyed. Can be empty.
    */
    void measureUpdateStep(const Vector& z_);

    /*! Returns the predicted state vector \f$ \tilde{x}_k \f$ (<em>a priori</em>
        state estimate). This function is used to predict a future state. 

        First, it calls \c sizeUpdate() to resizes any matrix that needs it. 
        Next, it calls a subset of the time update function - that is, it 
        only updates \a x and not P. This also means that only the 
        following virtual functions <em>will be</em> called :

        - \c makeCommonProcess() 
        - \c makeProcess()

        \param u_ Input vector. Will \b not be destroyed. Can be empty.
        \note The real \a x is not modified by this function (this is a 
        \c const function). Only a copy of \a x is returned.
        \warning For better efficiency, the prediction is returned by 
        reference.
        The reference points to an internal member of the filter, which means
        that a new prediction (and many other functions) will invalidate
        the contents of this vector.
        This also means that this vector must be copied (or better yet, 
        swapped) as soon as possible if its data is needed later.
    */
    const Vector& predict(Vector& u_);

    /*! Returns the predicted measurement vector \f$ \tilde{z}_k \f$.
        This function is used to predict a future measurement. 

        First, it calls \c sizeUpdate() to resizes any matrix that needs it.

        Then, it does a partial correction/measure update that is
        only \a z is calculated : \a x and P are not updated. This also 
        means that only the following virtual functions <em>will be</em>
        called :

        - \c makeCommonMeasure() 
        - \c makeMeasure().

        \note This is a \c const function. It only works on copies of vectors.
        \warning For better efficiency, the prediction is returned by 
        reference.
        The reference points to an internal member of the filter, which means
        that a new prediction (and many other functions) will invalidate
        the contents of this vector.
        This also means that this vector must be copied (or better yet, 
        swapped) as soon as possible if its data is needed later.
    */
    const Vector& simulate();

    //! Returns the corrected state \f$ \hat{x}_k \f$ (<em>a posteriori</em>
    //! state estimate).
    const Vector& getX() const;

    /*! Returns the <em>a posteriori</em> error covariance estimate matrix.
        \warning This is not a simple getter since P is not kept and updated
        in the filter (an alternate and more stable representation of P is 
        used), the value P must be calculated. Use this function wisely.
        \warning For better efficiency, P is returned by reference.
        The reference points to an internal member of the filter, which means
        that other functions may invalidate the contents of this matrix.
        This also means that this matrix must be copied (or better yet, 
        swapped) as soon as possible if its data is needed later.
    */
    const Matrix& calculateP() const;

    //! @}

  protected:

    /*! Allows optimizations on some calculations.
        By default, the EKFilter template class assumes that matrix 
        pre-creators and creators modify all matrices. However, if it 
        some of the functions do not modify anything, then some of the
        calculations could be optimized away. The \c NoModification() 
        function says that the function in which it has been called 
        has not modified any matrix. For optimization purposes, this 
        means that this function should be called in every non-mutating
        execution branch of all \c make*() and \c makeBase*() functions.
    */
    void NoModification();

    // TODO !!! : validate all virtual functions : can dims be 0 ?
    
    /*! \name Matrix Pre-Creators
        Theses functions have been designed to be overridden by derived classes
        if necessary. Their role is to fill in the parts of the %Kalman matrices
        that don't change between iterations. That is to say, these functions
        should only set constant values inside matrices that don't depend
        on \a x or \a u. 
       
        These are called once per resize (not necessarily at the moment of resize)
        including when the matrices are first allocated and are guaranteed to be
        called prior to their corresponding creator member functions.
        
        \note Matrices will have been properly resized before these functions 
        are called, so no further resizing is or should be necessary.
        \note If a matrix pre-creator is overridden, but it does not modify
        in any way the matrix in certain execution paths, then the function
        \c NoModification() should be called in each of those execution paths
        so that the filter can optimize away some calculations. The default
        versions of the matrix pre-creators simply call \c NoModification() and
        nothing else in their bodies.

        \warning Each matrix pre-creator cannot assume that any other matrix
        pre-creator will be called before or after it.
    */

    //! @{

    //! Virtual pre-creator of \a A.
    virtual void makeBaseA();

    //! Virtual pre-creator of \a W.
    virtual void makeBaseW();

    //! Virtual pre-creator of \a Q.

    /*! \note If \c OQ is \c true, that is, if \c Q is always diagonal, then
        it is not necessary to initialize non-diagonal elements with anything
        meaningful.
    */
    virtual void makeBaseQ();

    //! Virtual pre-creator of \a H.
    virtual void makeBaseH();

    //! Virtual pre-creator of \a V.

    /*! \note If \c OVR is \c true, that is, if \c both V and R are always 
        diagonal, then
        it is not necessary to initialize non-diagonal elements with anything
        meaningful.
    */
    virtual void makeBaseV();

    //! Virtual pre-creator of \a R.

    /*! \note If \c OVR is \c true, that is, if \c both V and R are always 
        diagonal, then
        it is not necessary to initialize non-diagonal elements with anything
        meaningful.
    */
    virtual void makeBaseR();

    //! @}

    /*! \name Matrix Creators
        Theses functions have been designed to be overridden by derived classes
        if necessary. Their role is to fill in the parts of the %Kalman matrices
        that change between iterations. That is to say, these functions
        should set values inside matrices that depend
        on \a x or \a u.
       
        These functions can assume that their corresponding matrix pre-creator
        has been called at least once before. Also, \c makeCommonProcess()
        will be called every time prior invoking \c makeA(), \c makeW(), 
        \c makeQ() and \c makeProcess(). Similarly \c makeCommonMeasure()
        will be called prior to \c makeH(), \c makeV(), \c makeR() and 
        \c makeMeasure().
        
        \note Matrices have already been properly resized before these 
        functions are called, so no further resizing is or should be necessary.
        \note If a matrix creator is overridden, but it does not modify
        in any way the matrix in certain execution paths, then the function
        \c NoModification() should be called in each of those execution paths
        so that the filter can optimize away some calculations. The default
        versions of the matrix creators only call \c NoModification() in
        their bodies.
        \warning Each matrix creator cannot suppose that any other matrix
        creator will be called before or after it. One thing is sure : 
        \c makeCommon*() is called first, then some of \c make*() and finally,
        \c makeProcess() or \c makeMeasure().
        \warning These functions can access \a x and \a u in read-only mode, 
        except makeProcess(), which must modify \a x.
    */

    //! @{

    /*! Optional function used to precalculate common values for process.
        If complex calculations are needed for more than one of \c makeA(),
        \c makeW(), \c makeQ() and \c makeProcess() functions, then this 
        function can be used to store the results in temporary variables 
        of the derived class.

        \warning This function must not modify any matrix of the base class.
        \warning This function must not be used to store permanent state. In 
        other words, all calculations performed in this function should be
        temporary. This is because the \c predict() function will call
        this function but has no knowledge of how to undo it.
    */
    virtual void makeCommonProcess();

    //! Virtual creator of \a A.
    virtual void makeA();

    //! Virtual creator of \a W.
    virtual void makeW();

    //! Virtual creator of \a Q.

    /*! \note If \c OQ is \c true, that is, if \c Q is always diagonal, then
        it is not necessary to initialize non-diagonal elements with anything
        meaningful.
    */
    virtual void makeQ();

    /*! Implements the process equation:

        \f[ \tilde{x}_k = f( \hat{x}_{k-1}, u_{k-1}, 0) \f]

        This function should assume that it is called with the data member 
        \b x holding the last corrected value \f$ \hat{x}_{k-1} \f$. When it exits,
        it should set the field \b x to be the new predicted value 
        \f$ \tilde{x}_{k} \f$.

        This function \b must be overridden since it is the core of the process 
        equation.

        \warning This function should have no side effects to class
        members (even members of derived classes) other than \a x. This is
        because this function is used by \c predict(), which does a calculation
        and then undoes it before returning the result.
    */
    virtual void makeProcess() = 0;
    
    /*! Optional function used to precalculate common values for measurement.
        If complex calculations are needed for more than one of \c makeH(),
        \c makeV(), \c makeR(), \c makeMeasure() and \c makeDZ()
        functions, then this function can be used to store the results in
        temporary variables of the derived class.
        \warning This function must not modify any matrix of the base class.
        \warning This function must not be used to store permanent state. In 
        other words, all calculations performed in this function should be
        temporary. This is because the \c simulate() function will call
        this function but has no knowledge of how to undo it.
    */
    virtual void makeCommonMeasure();

    //! Virtual creator of \a H.
    virtual void makeH();

    //! Virtual creator of \a V.
    /*! \note If \c OVR is \c true, that is, if \c both V and R are always 
        diagonal, then
        it is not necessary to initialize non-diagonal elements with anything
        meaningful.
    */
    virtual void makeV();

    //! Virtual creator of \a R.

    /*! \note If \c OVR is \c true, that is, if \c both V and R are always 
        diagonal, then
        it is not necessary to initialize non-diagonal elements with anything
        meaningful.
    */
    virtual void makeR();

    /*! Implements the measurement function:

       \f[ \tilde{z}_k = h \left( \tilde{x}_k, 0 \right) \f]

        This function \b must be overridden, since it is the core of the
        measurement equation. At the time this will be called, the data
        member \a x must contain the predicted state (i.e. <em>a priori</em> 
        state estimate) \f$ \tilde{x}_k \f$ which was set by a previous call
        to \c makeProcess(). 

        When the function returns, it must have set the data member \a z to
        be the \f$ \tilde{z}_k \f$ .

        \warning This function should have no side effects to class
        members (even members of derived classes) other than \a z. This is
        because this function is used by \c simulate(), which does a 
        calculation and then undoes it before returning the result.
    */
    virtual void makeMeasure() = 0;

    /*! Provides a (rarely if ever used) hook to allow a class to modify 
        the innovation vector (i.e. the real measurement vector \f$ z_k \f$
        minus the predicted measurement vector \f$ \tilde{z}_k \f$ ):

         \f[ \Delta{z}_k = ( z_k - \tilde{z}_k ) \f]

        which is stored in the data member \b dz.

        This function should rarely be overridden ; this is more of a hack than
        anything else. In fact, this is used to perform adjustments to 
        \f$ \Delta{z}_k \f$ . This is needed, for example, when the measurements
        include angles. It may be mandatory that the difference of the two 
        angles fall within a certain range, like \f$ [-\pi, \pi] \f$.
    */
    virtual void makeDZ();

    //! @}

    /*! Resizes all vector and matrices. \b Never call or overload this !
        \internal This function uses the \a flags bitfield to know
        which dimensions have changed since the last iteration. With this
        knowledge, only the needed matrices and vectors are resized.
        If it is also needed, the matrix P is factorized and stored in \a U.
    */
    virtual void sizeUpdate();

    //! \name %Kalman Vectors and Matrices
    //! @{

    /*! This is an \a n-sized vector. Derived classes should modify it only
        through \c makeProcess().
    */
    Vector x; //!< Holds either the previous corrected state value \f$ \hat{x}_{k-1}\f$, predicted state \f$ \tilde{x}_k \f$ or the new corrected state vector \f$ \hat{x}_k\f$.

    /*! This is an \a nu-sized vector. Derived classes should never 
        modify it.
    */
    Vector u; //!< Holds the control input vector \f$ u_{k-1} \f$ .

    /*! This is an \a m-sized vector. Derived classes should modify it only
        through \c makeMeasure().
    */
    Vector z; //!< Holds the predicted measurement vector \f$ \tilde{z}_k \f$.

    /*! This is an \a m-sized vector. Derived classes should modify it only
        through \c makeDZ(). The innovation vector is the difference between
        the real measurement vector \f$ z_k \f$ and the predicted one 
        \f$ \tilde{z}_k \f$.
    */
    Vector dz; //!< Holds the innovation vector \f$ \Delta{z}_k = z_k - \tilde{z}_k \f$.

    /*! This is an \a n by \a n jacobian matrix of partial derivatives, 
        defined as follow : 
        \f[ A_{[i,j]} = \frac{\partial f_{[i]}}{\partial x_{[j]}} \f]
        Derived classes should modify it only through \c makeBaseA() for
        the constant part and \c makeA() for the variable part.
    */
    Matrix A; //!< A jacobian matrix of \f$ f(x,0) \f$.

    /*! This is an \a n by \a nw jacobian matrix of partial derivatives, 
        defined as follow : 
         \f[ W_{[i,j]} = \frac{\partial f_{[i]}}{\partial w_{[j]}} \f]
         Derived classes should modify it only through \c makeBaseW() for
         the constant part and \c makeW() for the variable part.
    */
    Matrix W; //!< A jacobian matrix of \a \f$ f(\hat{x}_{k-1},w) \f$

    /*! This is the \a nw by \a nw covariance matrix of \a w, 
        that is : 
         \f[ Q = E\left( w w^T \right) \f]
        Derived classes should modify it only through \c makeBaseQ() for
        the constant part and \c makeQ() for the variable part.
        If \a Q is always diagonal, then you should turn on the \c OQ 
        optimization.
    */
    Matrix Q;           //!< Process noise covariance matrix.

    /*! This is an \a m by \a n jacobian matrix of partial derivatives, 
        defined as follow : 
        \f[ H_{[i,j]} = \frac{\partial h_{[i]}}{\partial x_{[j]}} \f]
        Derived classes should modify it only through \c makeBaseH() for
        the constant part and \c makeH() for the variable part.
    */
    Matrix H;    //!< A jacobian matrix of \f$ h(x, 0) \f$.

    /*! This is an \a m by \a nv jacobian matrix of partial derivatives, 
        defined as follow : 
        \f[ V_{[i,j]} = \frac{\partial h_{[i]}}{\partial v_{[j]}} \f]
        Derived classes should modify it only through \c makeBaseV() for
        the constant part and \c makeV() for the variable part.
        If both V and R are always diagonal, then you should turn on the
        \c OVR optimization.
    */
    Matrix V; //!< A jacobian matrix of \f$ h(\hat{x}_{k-1}, v)\f$.

    //! This is the \a nv by \a nv covariance matrix of \a v, 
    //! that is : 
    //! \f[ R = E\left( v v^T \right) \f]
    //! Derived classes should modify it only through \c makeBaseR() for
    //! the constant part and \c makeR() for the variable part.
    //! If both \a V and \a R are always diagonal, then you should turn on the
    //! \c OVR optimization.
    Matrix R;           //!< Measurement noise covariance matrix.

    //! @}

    //! \name %Kalman Dimensions
    //! \warning These values, which are accessible to derived classes, are 
    //! read-only. The derived classes should use the resizing functions
    //! to modify vector and matrix dimensions.
    //! @{

    K_UINT_32 n;        //!< Size of the state vector.
    K_UINT_32 nu;       //!< Size of the control input vector.
    K_UINT_32 nw;       //!< Size of the process noise vector.
    K_UINT_32  m;       //!< Size of the measurement vector.
    K_UINT_32 nv;       //!< Size of the measurement noise vector.

    //! @}

  private:
    
    //! Inplace upper triangular matrix Cholesky (UDU) factorization.

    //! This function is based on an algorithm in presented in appendix III.A in
    //! \ref refpage "[01]". It is used to transform \c P_ into \f$ U D U^T \f$.
    //! Quoting from \ref refpage "[01]" : "This mechanization is such that the 
    //! lower portion of \c P_ is not used and U and D can share the upper
    //! triangular portion of \c P_ (the diagonal elements of U are implicitly
    //! unity). In any case the upper triangular portion of P is destroyed
    //! by this mechanization."
    static void factor(Matrix& P_);

    //! Inplace upper triangular matrix inversion.

    //! This function calculates the inverse of \c P_ with an efficient
    //! algorithm, based on the fact that P_ is triangular. The result of
    //! the inversion is stored in a transposed form in the lower part of 
    //! \c P_.
    //! \param P_ Upper triangular matrix with unit diagonal.
    static void upperInvert(Matrix& P_);

    //! MWG-S orthogonalization algorithm for U-D time update.

    //! This function is based on an algorithm in presented in appendix VI.A
    //! in \ref refpage "[01]". It is used to generate a state prediction and
    //! to update \a U.
    void timeUpdate();

    //! U-D convariance factorization update.

    //! This function is based on an algorithm in presented in appendix V.A in
    //! \ref refpage "[01]". It is used to generate a corrected state prediction
    //! and to update \a U. It must be called once per measure, with the 
    //! corresponding values of \a H, \a V and \a R.
    //! \param dz New (whitened) measurement difference to incorporate.
    //! \param r Covariance (whitened) of the measure.
    void measureUpdate(T dz, T r);

    //! \name Template Methods
    //! These are all template methods (in a design pattern sense, these are
    //! not template member functions). They simply call their corresponding
    //! virtual not-Impl functions, but adding some logic to take into account
    //! the \c NoModification() optimization.
    //! @{

    //! \c makeBaseA() template method.
    void makeBaseAImpl();

    //! \c makeBaseW() template method.
    void makeBaseWImpl();

    //! \c makeBaseQ() template method.
    void makeBaseQImpl();

    //! \c makeBaseH() template method.
    void makeBaseHImpl();

    //! \c makeBaseV() template method.
    void makeBaseVImpl();

    //! \c makeBaseR() template method.
    void makeBaseRImpl();

    //! \c makeA() template method.
    void makeAImpl();

    //! \c makeW() template method.
    void makeWImpl();

    //! \c makeQ() template method.
    void makeQImpl();

    //! \c makeH() template method.
    void makeHImpl();

    //! \c makeV() template method.
    void makeVImpl();

    //! \c makeR() template method.
    void makeRImpl();

    //! @}

    //! This matrix is the upper triangular Cholesky factorization of P. So, it
    //! should be a \a n by \a n matrix, but because of algorithmic issues,
    //! it is in fact a \a n by \a nn matrix. Usually, the factorization
    //! would yield two matrices, U and D ( \f$ P = U D U^T \f$ ), where
    //! U is an upper triangular matrix with unit diagonal, and D is a diagonal
    //! matrix. Since the unit diagonal is implicit in our representation,
    //! this matrix contains D on its diagonal, U in its upper part and junk
    //! in its lower part. This is for the left \a n by \a n part of the 
    //! matrix. For the right \a n by \a nw part, it is mainly junk, but
    //! it is used temporarily to hold a copy of \a W.
    Matrix U;           //!< Cholesky factorization of P.

    //! If \a Q is not diagonal, then process noise is correlated, and must
    //! be whitened for the algorithms to work. To achieve this result, we
    //! factorize \a Q like this : \f$ Q = U_q D_q U_q^T \f$. We then replace
    //! \a W by \a W_ ( \f$ = W U_q \f$ ) and \a Q by \a Q_ ( \f$ = D_q \f$ ).
    Matrix W_;          //!< Modified version of \a W to whiten process noise.

    //! If \a Q is not diagonal, then process noise is correlated, and must
    //! be whitened for the algorithms to work. To achieve this result, we
    //! factorize \a Q like this : \f$ Q = U_q D_q U_q^T \f$. We then replace
    //! \a W by \a W_ ( \f$ = W U_q \f$ ) and \a Q by \a Q_ ( \f$ = D_q \f$ ).
    Matrix Q_;          //!< Modified version of \a Q to whiten process noise.

    //! If \a V and \a R are not both diagonal, then \f$ V R V^T \f$ if not
    //! diagonal : measurement noise is 
    //! not normalized, and must be modified for the algorithms to work. To 
    //! achieve this result, we factorize it like this : 
    //! \f$ V R V^T = U_r D_r U_r^T \f$. We then replace \f$ V R V^T \f$ by
    //! \a R_ \f$ ( = D_r ) \f$, 
    //! \a H by \a H_ ( \f$ = U_r^{-1} H \f$ ) and 
    //! \a dz by \a _x ( \f$ = U_r^{-1} dz \f$ ).
    Matrix H_;          //!< Modified version of \a H to whiten measurement noise.

    //! If \a V and \a R are not both diagonal, then \f$ V R V^T \f$ if not
    //! diagonal : measurement noise is 
    //! not normalized, and must be modified for the algorithms to work. To 
    //! achieve this result, we factorize it like this : 
    //! \f$ V R V^T = U_r D_r U_r^T \f$. This matrix contains the result of
    //! this factorization : the diagonal of \a R_ is \f$ D_q \f$, the upper
    //! part is \f$ U_q \f$ (the unit diagonal is implied) and the lower
    //! part is \f$ \left( U_q^{-1} \right)^T \f$ (the unit diagonal is 
    //! again implied).
    //!
    //! If both \a V and \a R are diagonal, then \f$ V R V^T \f$ is
    //! diagonal. In that case, \a R_ is in fact \f$ V R V^T \f$.
    Matrix R_;          //!< Modified version of \a R to whiten measurement noise.

    Vector a;           //!< Temporary vector.
    Vector d;           //!< Temporary vector.
    Vector v;           //!< Temporary vector.

    //! In fact, \f$ nn = n + nw \f$, so that \a U can contain \a W is its
    //! right part.
    K_UINT_32 nn;       //!< Number of columns of \a U

    mutable Matrix _P;  //!< Temporary matrix.
    mutable Vector _x;  //!  Temporary vector.

    K_UINT_16 flags;    //!< Bitfield keeping track of modified matrices.
    bool modified_;     //!< Boolean flag used by \c NoModification().

  };

}

#include "kalman/ekfilter_impl.hpp"

#endif

/*! \page Example
    \anchor expage
    \section example How to Use this Extended Kalman Filter Library?
    \subsection introduction Introduction
    This Extended %Kalman Filter library is powerful and very simple to use,
    but a %Kalman filter is very difficult to debug. So, it is very important
    to follow a procedure to be sure that everything is right (code and 
    equations). This example suggests a procedure to follow and shows how
    to use the library.  If you are not familiar with the %Kalman filter,
    please read this article [02].
   
    \subsection step1 Step 1 : Determine the system's mathematical model
    The first thing to do is to define the state vector you want to 
    estimate \f$ \vec x \f$ and the inputs \f$ \vec u \f$ of the system.
    After this, determine the non-linear process function \f$ f \f$ 
    that describes the evolution of the state vector through time, that is :

    \f[ \vec x_k = f \left( x_{k-1}, u_{k-1}, w_{k-1} \right) \f]

    where \f$ w \f$ is the process noise vector due to uncertainty and 
    process modeling errors.
    
    Finally, determine the non-linear relation between your state vector 
    \f$ \vec x \f$ and the measurement vector \f$ \vec z \f$.

    \f[ \vec z_k = h \left( x_{k-1}, v_{k-1} \right) \f]

    where \f$ v \f$ is the measurement noise vector.
   
    \par In this example :
   
    A plane flies in a 2D space where the x axis is the distance traveled
    by the plane and y axis is its altitude.  This system can be represented
    by the following continuous equations:
   
    \f[ \ddot x = \frac{u}{m} - \frac{b_x}{m} \dot x^{2} \f]
    \f[ \ddot y = \frac{p}{m} \dot x^{2} - g \f]
   
    where \f$ m \f$ is the plane's weight (1000 kg)\n
          \f$ b_{x} \f$ is the drag coefficient (0.35 N/m^2/s^2)\n
          \f$ p \f$ is the lift force (3.92 N/m^2/s^2)\n
          \f$ g \f$ is the gravitational acceleration (9.8 m/s^2)\n
          \f$ u \f$ (the input) is the motor's thrust\n
          \f$ \dot x \f$ is the first order derivative of x wrt time \f$ \frac{dx}{dt} \f$ \n
          \f$ \ddot x \f$ is the \f$ 2^{nd}\f$ order derivative of x wrt time \f$ \frac{d^2 x}{dt^2} \f$ \n
   

    The discrete equation in which the \f$ \vec x_k \f$ is determine in terms
    of the \f$ _{k-1} \f$ values is:

   \f[\vec x_k = \left [ \begin{array}{c} x_k \\ \\ \dot x_k \\ \\ y_k \\ \\ \dot y_k \end{array} \right] = 
  \left [ \begin{array}{c} 
   x_{k-1} + T \dot x_{k-1} + \frac{T^{2}}{2} \left( \frac{u}{m} - \frac{b_x}{m} \dot x_{k-1}^{2} \right )\\ \\
  \dot x_{k-1} + T \left ( \frac{u}{m} - \frac{b_x}{m} \dot x_{k-1}^{2} \right ) + w_{1} \\ \\
  y_{k-1} + T \dot y_{k-1} + \frac{T^{2}}{2} \left( \frac{p}{m} \dot x_{k-1}^{2} - g \right )\\ \\
  \dot y_{k-1} + T \left( \frac{p}{m} \dot x_{k-1}^{2} - g \right ) + w_{2}
  \end{array} \right ] \f]

    where \f$ w_{1} \f$ and \f$ w_{2} \f$ are the random variables which represent the process noise.
   
    A station on the ground (at the origin) mesures the angle between the
    plane and the ground (x axis) and the distance between the plane and the station.
    These measurements are based on the following equation:
   
    \f[\vec z_k = \left [ \begin{array}{c} \theta \\ \\ r \end{array} \right ] = 
    \left [ \begin{array}{c} atan(\frac{y}{x}) +v_{1} \\ \\ \sqrt{x^{2}+y^{2}} + v_{2} \end{array} \right ] \f]
   
    where \f$ v_{1} \f$ and \f$ v_{2} \f$ are the random variables which represent the process noise.
   
    \subsection step2 Step 2 : Calculate Jacobian matrix
    Calculate the jacobian matrix A, W, H and V where :
    A is an \a n by \a n jacobian matrix of partial derivatives, 
    defined as follow : 
    \f[ A_{[i,j]} = \frac{\partial f_{[i]}}{\partial x_{[j]}} \f]
   
    W is an \a n by \a nv jacobian matrix of partial derivatives, 
    defined as follow : 
    \f[ W_{[i,j]} = \frac{\partial f_{[i]}}{\partial w_{[j]}} \f]
   
    H is an \a m by \a n jacobian matrix of partial derivatives, 
    defined as follow : 
    \f[ H_{[i,j]} = \frac{\partial h_{[i]}}{\partial x_{[j]}} \f]
   
    V is an \a m by \a nv jacobian matrix of partial derivatives, 
    defined as follow : 
    \f[ V_{[i,j]} = \frac{\partial h_{[i]}}{\partial v_{[j]}} \f]
   
    \f$ n \f$ is the number of element in state vector\n
    \f$ m \f$ is the number of measure\n
    \f$ nw \f$ is the number of process noise random variables\n
    \f$ nv \f$ is the number of measurement noise random variables\n
   
    \par In this example:
   
    \f[ A = \left [ \begin{array}{cccc} 
    1 & T-T^{2} \frac{b}{m} \dot x_{k-1} & 0 & 0 \\ \\
    0 & 1-2T \frac{b}{m} \dot x_{k-1} & 0 & 0 \\ \\
    0 & T^{2} \frac{p}{m} \dot x_{k-1} & 1 & T \\ \\
    0 & 2T \frac{p}{m} \dot x_{k-1} & 0 & 1
    \end{array} \right ] \f]
   
    \f[ W = \left [ \begin{array}{cc} 
    0 & 0 \\ \\
    1 & 0 \\ \\
    0 & 0\\ \\
    0 & 1
    \end{array} \right ] \f]
   
    \f[ H = \left [ \begin{array}{cccc} 
    \frac{- \dot y_{k-1}}{x_{k-1}^{2}+y_{k-1}^{2}} & 0 \frac{ \dot x_{k-1}}{x_{k-1}^{2}+y_{k-1}^{2}} & 0 \\ \\
    \frac{ \dot x_{k-1}}{\sqrt{x_{k-1}^{2}+y_{k-1}^{2}}} & 0 \frac{ \dot y_{k-1}}{\sqrt{x_{k-1}^{2}+y_{k-1}^{2}}} & 0
    \end{array} \right ] \f]
   
    \f[ V = \left [ \begin{array}{cc} 
    1 & 0 \\ \\
    0 & 1
    \end{array} \right ] \f]
   
    \subsection step3 Step 3 : Initial conditions and covariance matrix
   
    Set initial estimation of the state vector. After, set the covariance matrix P 
    which represents the covariance of the error of the state vector estimation. 
    Then, set the covariance Q and R which represent the covariance matrix
    of process noise and measurement noise, respectively.
   
    \par In this example:
   
    The first estimation of the state vector is based on the first measurements
    and the covariance matrix are the following:
   
    \f[\vec x = \left [ \begin{array}{c} r\cos\theta \\ 60 \\ r\sin\theta \\ 0 \end{array} \right] \f]
   
    \f[ P = \left [ \begin{array}{cccc} 
    100^2 & 0 & 0 & 0 \\ \\
    0 & 10^2 & 0 & 0 \\ \\
    0 & 0 & 25^2 & 0 \\ \\
    0 & 0 & 0 & 10^2 \\ \\
    \end{array} \right ] \f]
   
    \f[ Q = \left [ \begin{array}{cc} 
    0.01^{2} & 0.01^{2}/10 \\ \\
    0.01^{2}/10 & 0.01^{2}
    \end{array} \right ] \f]
   
    \f[ R = \left [ \begin{array}{cc} 
    0.01^{2} & 0 \\ \\
    0 & 0.01^{2}
    \end{array} \right ] \f]
   
    \subsection step4 Step 4 : Implementation of the Kalman filter
   
    Now, it's time to create the first version of your %Kalman filter. 
    You should not try to optimize it at this step, just create your filter and validate it.
    This library allows you to optimize your filter, but in your first implementation, 
    code only the basic functions.  So, code functions called \c makeProcess(), \c makeMeasure(), 
    \c makeA(), \c makeH(), \c makeQ(), \c makeR(), \c makeV() and \c makeW(). These functions will set the value of each matrix.
    
    The first thing to do is to create your %Kalman filter class.
    
    \dontinclude simple_plane.h
    \skip class
    \until }

    In this example, our %Kalman filter inherits from the Extended %Kalman Filter, because it's a non-linear problem (\f$ f \f$ and \f$ h \f$ are non-linear functions)
    The first two template parameters are respectively the floating point type used by the filter (\c float or \c double) and the beginning index
    of vectors and matrices (0 or 1). There are three other template parameters to the \c EKFilter template class. 
    They are explained in the next section, but they
    can be safely set to their default values in the first version of the filter, which are false, false and true to 
    disable optimizations and enable bound-checking.
    
    You should declare each of the previously named functions, \c makeProcess(),
    \c makeMeasure() etc in this class. You can also declare the variables.
    
    Next, write the class constructor. You can call the function \c setDim() 
    here or you will call it manually in your \c main() function after you 
    create the filter object.  The function \c setDim() sets the number of 
    states, the number of inputs, the number of process noise random variables,
    the number of measurements and the number of measurement noise random 
    variables. It can be used by advanced users to implement a 
    Variable-Dimension Extended %Kalman Filter (an EKF whose dimensions 
    may change from one iteration to the other).
   
    \dontinclude simple_plane.cpp
    \skip cSimplePlaneEKF::cSimplePlaneEKF
    \until }

    In the function \c makeProcess(), you should use a temporary vector to 
    store the new state vector like this :
   
    \skip cSimplePlaneEKF::makeProcess
    \until }
   
    In the function \c makeMeasure(), you update directly the measurement 
    vector \f$ \vec z \f$.  These are the predicted measurements.
   
    \skip cSimplePlaneEKF::makeMeasure
    \until }
   
    Then, you code all other functions \c makeX() like this:
   
    \dontinclude simple_plane.cpp
    \skip cSimplePlaneEKF::makeA
    \until }
   
    \skip cSimplePlaneEKF::makeW
    \until }
   
    \skip cSimplePlaneEKF::makeQ
    \until }
   
    \skip cSimplePlaneEKF::makeH
    \until }
   
    \skip cSimplePlaneEKF::makeV
    \until }
   
    \skip cSimplePlaneEKF::makeR
    \until }
   
    Now, your filter is ready to be used.  In this example, the measurements
    and the inputs have been calculated by the \c generation.m Matlab script.
    It's a good idea to test your filter with known measurements and inputs if 
    you want to validate it.
   
    After you create the filter object, you should call the \c setDim() 
    member function before calling the \c init() function. In this example,
    the \c setDim() function is called in the class constructor.  The 
    \c init() function sets the initial state and the initial covariance
    matrix. 
   
    WARNING : The vectors passed to the \c init() function become unusable when \c init() returns ! Never use those vectors after the call.
   
    \dontinclude example.cpp
    \skip cPlaneEKF
    \until };
   
    \skip Initial
    \until init
   
    \skip for
    \until }
   
    Call the function \c step() for each iteration and pass the new inputs 
    and the new measurements.
   
    \subsection step5 Step 5 : Optimization
   
    When your %Kalman filter works properly, you can optimize it in many simple ways.
    
    -# If matrix Q is always diagonal, set the OQ template parameter to \c true. This will
    minimize some calculations. Also, you will only need to fill in diagonal elements of Q,
    since the other values will never be read.
    -# If both matrices V and R are always diagonal, set the OVR template parameter to \c true. This will
    minimize some calculations. Also, you will only need to fill in diagonal elements of V and R,
    since the other values will never be read.
    -# If some matrix values are constant, use the \c makeBaseX() function instead of the \c makeX() function
    to fill these values.
    These functions are called only once at the beginning instead of once per iteration. You can used 
    the \c makeBaseX() function to set values that never change in a matrix and just set the other
    values in the \c makeX() function.
    -# If complex calculations are needed for more than one of \c makeA(), \c makeW(), \c makeQ() and \c makeProcess()
    functions, then use the function \c makeCommonProcess() to do those calculations and save them in member variables of your own subclass.  This function is
    always called before the others.
    -# If complex calculations are needed for more than one of \c makeH(), \c makeV(), \c makeR(), \c makeMeasure()
    and \c makeDZ() functions, then use the function \c makeCommonMeasure() to do those calculations and save them in member variables of your own subclass.  This function is
    always called before the others.
    -# While writing a \c makeX() or a \c makeBaseX() function, there can be some execution paths where the function
    does not modify any matrix (for example, if there is some condition, then modify the matrix, else don't).
    If this is the case, then each non-mutating execution path should call \c NoModification() before returning,
    so that some calculations can be avoided.
    -# When your filter works properly, set the debug template parameter to \c false to disable bound-checking.
   
    \par In this example:
   
    This example have been optimized in many ways:
   
    - The matrices V and R are diagonal, so the OVR template parameter is set to true.  Then, we just have to 
    set the diagonal values of those matrices.
    - The matrices V, R, W and Q never change, so we use \c makeBaseX() functions instead of \c makeX() functions.
    - Constants values of A and H have been moved from the \c makeX() functions to \c makeBaseX() functions.
    - The filter works properly, so we don't need bound-checking. Let's set the debug template parameter to \c false.
   
    So, the final result for this example is:
   
    \include plane.h
   
    \include plane.cpp
   
    \section reference References
    [01] Bierman, G. J. "Factorization Methods for Discrete Sequential
    Estimation", Academic Press, 1977. \n
    [02] Welch, G. and Bishop, G. "An Introduction to the %Kalman Filter",
    http://www.cs.unc.edu/~welch/kalman/kalmanIntro.html
   

    \mainpage The KFilter Project: A Variable Dimension Extended Kalman Filter Library
    \anchor mainpage

    "The %Kalman filter is a set of mathematical equations that provides an
    efficient computational (recursive) solution of the least-squares method.
    The filter is very powerful in several aspects: it supports estimations
    of past, present, and even future states, and it can do so even when the
    precise nature of the modeled system is unknown." (quoted from [02])

    This version of the %Kalman filter is a Variable-Dimension Extended 
    %Kalman Filter (VDEKF). It supports optimized algorithms 
    (translated from Fortran - see [01]), even in the 
    presence of correlated process or measurement noise.

    For a usage example, please check out the \ref expage "example page"
    in the related pages section.

    \par Notation

    We use the following notation taken from [02]. We assume
    that our system evolves over time in discrete time steps. In each step or
    iteration, there will be two phases:

    - a prediction phase where we have a last system state \f$ x_{k-1} \f$
    and potentially a control signal input \f$ u_{k-1} \f$ . This equation
    will tell us the new state of the system given that we know its previous
    state. If there is also the optional control signal input, that input 
    represents some input that we might supply to our system to affect its
    behavior. E.g. opening the throttle to an airplane's engine. Our equation
    should take that into account as well to give us the predicted new state.
    Finally, this equation will take ito account noise associated with our 
    process e.g. turbulence associated with an airplane will cause the altitude
    of the plane to change unexpectedly. These 3 factors will be used to
    predict the new state of the system.
    - a correction phase where upon completion we want the best "guess" as
    to what state the system is in. To do this, we first need to take
    measurements (sensor readings) as to what is the system's present state.
    The %Kalman filter will use the measurements taken in conjunction with
    the predicted state to estimate the new "corrected" state. Hence the 
    calculation of this corrected state will use the predicted state, the
    measurements and the process and measurement noise to give a best "guess"
    as to what the system's new state will be. This best guess is the 
    "corrected" state.

    For bookeeping, we'll identify vectors according to which step or 
    iteration that they were measured or calculated. E.g. for each 
    step 0, 1, ... k-1, k , we say that for a vector \b v , 
    that \f$ v_{i} \f$ is the value of the vector v for the 
    \f$ i^{th} \f$ step.

    To distinguish between the system's real state, the predicted state,
    and the corrected state, we'll use the following symbols to represent the
    state vector:

    - \f$ x \f$ represents the system's real state. It is purely for
    discussion only. We have no real way of knowing what this state is while
    for the control signal input vector \f$ u \f$, we do.
    - \f$ \tilde{x} \f$ represents the predicted state based on
    the previously corrected state and a control signal \f$ u \f$ telling us the
    new desired state. \f$ \tilde{x} \f$ is where we predict what the system 
    will end up after applying the control signal \f$ u \f$.
    - \f$ \hat{x} \f$ represents a corrected state obtained by weighing
    the actual measured state with our predicted state. This is done because
    we can't fully trust either the predicted state (due to its process noise)
    nor the measured state (due to its measurement noise). This state is our
    probable or estimated state of the system.

    \note Many texts use the \f$ \hat{x}_k^- \f$ to represent the apriori
    estimate rather than \f$ \tilde{x}_k \f$ .

    The ekfilter supports the more general case where the process equations 
    are non-linear. 

    \par For the prediction phase of our step,

    let \f$ f \f$ be a non-linear process function that describes the 
    evolution of the system's state through time. Given the previous iteration's
    system state \f$ \hat{x}_{k-1} \f$, \b f will be used to predict what the 
    new state \f$ \tilde{x} \f$ will be if we apply a control signal \b u to 
    the system and if the process has some noise modeled by \b w, that is :

    \f[ \tilde{x}_k = f \left( \hat{x}_{k-1}, u_{k-1}, w_{k-1} \right) \f]

    where
      - \f$ f \f$ is the known non-linear process function describing our system
      - \f$ \hat{x}_{k-1} \f$ is the vector representing the system's 
        corrected state as obtained from the previous step. Recall that \b x 
        is what is the system's state that we want to estimate based on a 
        weighted combination of the predicted state based upon our theoretical
        process calculations and the noisy measurements reported by our sensors.
      - \f$ \tilde{x}_{k} \f$ is the vector representing the what we predict 
        the system's state will be once we apply the control signal 
        \f$ u_{k-1} \f$ i.e. this is the predicted state for the system 
        after \f$ u_{k-1} \f$ is applied. \f$ \tilde{x}_k \f$ can be thought 
        of as the system's state based on our theoretical model with some
        process noise thrown in.
      - \f$ u_{k-1} \f$ is the (known) control input vector fed to the system
        process to change the system's state.
      - \f$ w_{k-1} \f$ is the (unknown) process noise vector due to process
        errors. An example of process noise would be aircraft turbulence and
        how it affects an airplane's altitude. It is not something predictable
        and is inherit part of the process.

    Further suppose that the (known) process noise covariance matrix is :

    \f[ Q = E \left( w w^T \right) \f]

    \par For the correction phase of the step, 

    we need to model how we will determine what the real state of the
    system is. So let's assume we have a (known) measurement vector
    \f$ z \f$ (e.g. some sensor outputs), which depends on the system's
    current real state \f$ x_{k} \f$ in the form of a non-linear 
    function \f$ h \f$ . This equation would look like:

    \f[ \tilde{z}_k = h \left( \tilde{x}_k, v_k \right) \f]

    where \f$ v \f$ is the (unknown) measurement noise vector with
    a (known) covariance matrix :

    \f[ R = E \left( v v^T \right) \f]

    Examples of measurement noise are found in real world sensors. Any physical
    sensor will have some degree of inaccuracy. 

    Finally, there is the need to determine our best guess as to what the 
    corrected state \f$ \hat{x}_k\f$ should be for the k<sup>th</sup> step.
    This will be based on the predicted value \f$ \tilde{x}_k \f$ and the 
    last measured value \f$ z_k \f$ and is given by the equation:

    \f[ \hat{x}_k = \tilde{x}_k + K_k \left( z_k - \tilde{z}_{k} \right) \f]
    
    where K<sub>k</sub> is the %Kalman gain. Now the remaining question is to
    how we determine the %Kalman gain. The %Kalman equations need to determine
    how much noise is in the data for this particular iteration. To do this,
    it looks at the difference between what the measurements should 
    theoretically be (i.e. \f$ \tilde{z}_k \f$ and the actual measurement
    \f$ z_k \f$. This with the covariance matrices allow it to determine the
    best corrected measurement for a linear system.

    While the %Kalman algorithm requires using linear equations, we're dealing
    with non-linear equations \b f and \b h. For determining the %Kalman gain,
    we need to use linear equations. To do this, we'll use linear approximations
    for the \b f and \b h functions around the place where we expect the state 
    to be. In this way, hopefully our linear approximations won't be too far off.

    \par Linearizing the non-linear equations for use in the Filter

    Suppose that we have a corrected estimate of the previous state 
    \f$ \hat{x}_{k-1} \f$, called a corrected state or an <em>a posteriori</em>
    state estimate for the k-1<sup>th</sup> iteration. We can build a predicted
    state (also called an <em>a priori</em> state estimate) for the new 
    k<sup>th</sup>iteration by using \f$ f \f$ :

    \f[ \tilde{x}_k = f \left( \hat{x}_{k-1}, u_{k-1}, 0 \right) \f]

    We set the process noise to be 0 as its actual value is unknown.
    With this theoretically predicted state, we can get a predicted
    measurement vector (sensor outputs) by using \f$ h \f$ :

    \f[ \tilde{z}_k = h \left( \tilde{x}_k, 0 \right) \f]

    Again the measurement noise is set to 0 since it is also unknown.

    Now we know the corrected state \f$ \hat{x}_k \f$ will be close
    to \f$ \tilde{x}_k \f$ and the actual sensor measurements \f$ z_k \f$ will
    be close to \f$ \tilde{z}_k \f$ . We need some linear matrices to allow
    us to work with the %Kalman filter.

    To obtain a linear, least-squares formulation, we need to linearize
    \b f and \b g. We use the first-order Taylor series centered on 
    the previous measured state \f$ \hat{x}_{k-1} \f$ to get an equation
    to predict our new state \f$ \tilde{x}_{k} \f$:

    \f[ x_k \approx 
         f \left( \hat{x}_{k-1}, u_{k-1}, 0 \right)
        + \frac{\partial f}{\partial x} \left( \hat{x}_{k-1}, u_{k-1}, 0 \right)
      \left( \Delta x \right)
    + \frac{\partial f}{\partial u} \left( \hat{x}_{k-1}, u_{k-1}, 0 \right)
      \left( \Delta u \right)
    + \frac{\partial f}{\partial w} \left( \hat{x}_{k-1}, u_{k-1}, 0 \right)
      \left( \Delta w \right) \f]
    \f[ \phantom{x_k} = 
    \tilde{x}_k + A \left( x_{k-1} - \hat{x}_{k-1} \right) + W w_{k-1} \f]

    Since we are not changing the control signal \f$ \hat{u}_{k-1} \f$,
    that term drops out leaving us with just terms involving 
    \f$ {x}_{k-1} \f$ the system state and \f$ w_{k-1} \f$ the process noise.

    We can do the same for the other equation :

    \f[ z_k \approx h \left( \tilde{x}_k, 0 \right) 
    + \frac{\partial h}{\partial x} \left( \tilde{x}_k, 0 \right)
      \left( \Delta x \right)
    + \frac{\partial h}{\partial v} \left( \tilde{x}_k, 0 \right)
      \left( \Delta v \right) \f]
    \f[ \phantom{z_k} = 
    \tilde{z}_k + H \left( x_k - \tilde{x}_k \right) + V v_k \f]

    \par Final equations
    The non-linear equations that determine our predicted values:

    \f[ \tilde{x}_k = f \left( \hat{x}_{k-1}, u_{k-1}, w_{k-1} \right) \f]
    \f[ \tilde{z}_k = h \left( \tilde{x}_k, v_k \right) \f]

    And our linear equations that are used to calculate the %Kalman gain
    \f$ K_k \f$ so that we can use the equation:

    \f[ \hat{x}_k = \tilde{x}_k + K_k \left( z_k - \tilde{z}_k \right) \f]

    to determine the new corrected state \f$ \hat{x}_k \f$ . These linear 
    equations are:

    - \f$ x_k = \tilde{x}_k + A \left( x_{k-1} - \hat{x}_{k-1} \right) + W w_{k-1} \f$
    - \f$ z_k = \tilde{z}_k + H \left( x_k - \tilde{x}_k \right) + V v_k \f$
    - \f$ Q = E \left( w w^T \right) \f$
    - \f$ R = E \left( v v^T \right) \f$

    where

    - A is the jacobian matrix of \f$ f(x,\hat{u}_{k-1},0) \f$.
    - W is the jacobian matrix of \a \f$ f(\hat{x}_{k-1},w) \f$
    - Q is the process noise covariance matrix.
    - H is the jacobian matrix of \f$ h(x, 0) \f$.
    - V is the jacobian matrix of \f$ h(\hat{x}_{k-1}, v)\f$.
    - R is the measurement noise covariance matrix.
    - \f$ \hat{x}_{k-1} \f$ is the corrected state of the previous iteration k.
    - \f$ \tilde{x}_k \f$ is the predicted state for the current iteration k.
    - \f$ u_{k-1} \f$ is the control signal for the previous iteration k - 1 .
    - \f$ w_{k-1} \f$ is the previous process noise.
    - \f$ v_{k-1} \f$ is the previous measurement noise.
    - \f$ \tilde{z}_k \f$ is the predicted measurement for this iteration based on the predicted state.
    - \f$ z_k \f$ is the actual measurement
    - \f$ \Delta z = z_k - \tilde{z}_k \f$ is the innovation vector also known as the residual.

    \note If \f$ P_{k} \f$ is the covariance of the prediction process at
    iteration k, then we have \f$ P_{k} = AP_{k-1}A^T + Q \f$.

    This is all that we need to supply to the EKalmanFilter. The the above will
    provide enough information for it to calculate the %Kalman gain to
    to give us the corrected value.

    See the \ref templateEKFilter "EKFilter" and the \ref templateKFilter "KFilter" .

    \par References
    \anchor refpage
    [01] Bierman, G. J. "Factorization Methods for Discrete Sequential
    Estimation", Academic Press, 1977. \n
    [02] Welch, G. and Bishop, G. "An Introduction to the %Kalman Filter",
    http://www.cs.unc.edu/~welch/kalman/kalmanIntro.html
*/
